import os
import io
import time
import heapq
import openai
import base64
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as T

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Any, Literal, Iterable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.utils import save_image
from packaging import version
from functools import partial

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    from torch.amp.grad_scaler import GradScaler
    from torch.amp.autocast_mode import autocast

    cuda_autocast = partial(autocast, device_type="cuda")
else:
    from torch.cuda.amp import GradScaler, autocast

    cuda_autocast = autocast

from configs.types import TrainerConfigs
from arch.data_module import DataModule
from arch.loss_module import LossModule
from utils.loggers import get_text_logger, get_train_logger
from utils.charset_adapter import CharsetAdapter


class Trainer:

    def __init__(self, configs: TrainerConfigs):

        self.ngpu = configs.ngpu
        self.num_epochs = configs.num_epochs
        self.init_lr = configs.init_lr
        self.loss_type = configs.loss_type

        self.world_size = os.environ["WORLD_SIZE"]
        self.local_rank = os.environ["LOCAL_RANK"]

        # Loggers
        self.enable_log = self.local_rank == 0
        self.text_logger = get_text_logger(__name__) if self.enable_log else None
        self.writer = get_train_logger(Path(configs.log_dir)) if self.enable_log else None
        self.save_dir = Path(configs.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _record_text_log(self, method: str, msg: str):
        if self.enable_log:
            self.text_logger.__getattribute__(method)(msg)

    @torch.no_grad()
    def _record_scalar(self, name: str, value: float, step: int):
        if self.enable_log:
            pass

    def _to_device(self, inputs: tuple) -> tuple:
        return inputs

    def _finish_train(self):
        self.writer.close()

    def _train_batch(model: nn.Module, train_batch: tuple[Any], criterion) -> float:
        with cuda_autocast:
            outputs = model.train(train_batch)
        # CTC loss must be float32 for numerical stability
        loss = criterion(outputs)
        return loss

    @torch.inference_mode()
    def _val_batch(model: nn.Module, val_batch: tuple[Any], val_criterion) -> float:
        outputs = model.val(val_batch)
        metric = val_criterion(outputs)
        return metric

    @torch.no_grad()
    def _save_checkpoint(self, file_path, model, optimizer, scaler, epoch, metric_name, metric):
        if not self.enable_log:
            return
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.module.state_dict() if self.ngpu > 1 else model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
            },
            file_path,
        )

    @torch.no_grad()
    def _save_topk(
        self,
        top_models: list[tuple[float, int, Path]],
        accuracy: float,
        epoch_count: int,
        model: nn.Module,
        optimizer,
        scaler,
        topk: int = 3,
    ):
        self._save_checkpoint(
            self.save_dir / "lastest.pth", model, optimizer, scaler, epoch_count, "accuracy", accuracy
        )
        file_path = self.save_dir / f"checkpoint-epoch-{epoch_count}-accuracy-{accuracy:.4f}.pth"
        if len(top_models) < topk:
            heapq.heappush(top_models, (accuracy, epoch_count + 1, file_path))
            self._save_checkpoint(file_path, model, optimizer, scaler, epoch_count)
        else:
            if accuracy > top_models[0][0]:
                removed_acc, removed_epoch, removed_path = heapq.heappop(top_models)
                if os.path.exists(removed_path):
                    os.remove(removed_path)
                heapq.heappush(top_models, (accuracy, epoch_count + 1, file_path))
                self._save_checkpoint(file_path, model, optimizer, scaler, epoch_count)
        return top_models

    @torch.no_grad()
    def _load_data_batch(
        self, data_module: DataModule, stage: Literal["train", "trainedit", "test", "val"]
    ) -> Iterable[tuple[Any]]:
        dataset: Dataset = data_module.__getattribute__(f"{stage}set")
        dataset = ConcatDataset(dataset.values())
        dataloader: DataLoader = data_module.__getattribute__(f"{stage}_loader")(dataset, ddp=self.ngpu > 1)
        yield next(iter(dataloader))

    @torch.inference_mode()
    def val(self, model: nn.Module, data_module: DataModule, validator) -> None:

        # Datasets & DataLoader
        valsets = data_module.valset
        valsets = ConcatDataset(valsets.values())
        val_loader = data_module.val_loader(valsets, ddp=self.ngpu > 1)

        # Model
        model = DDP(model, device_ids=[self.local_rank])
        model.eval()

        val_criterion = validator
        avg_acc = []

        for epoch_count in range(self.num_epochs):
            self._record_text_log("info", f"Epoch {epoch_count + 1} / {self.num_epochs}")
            val_pbar = tqdm(
                val_loader,
                desc=f"Eval",
                colour="yellow",
                disable=not self.enable_log,
            )
            for val_batch in val_pbar:
                val_batch = self._to_device(val_batch)
                accuracy: float = self._val_batch(model, val_batch, val_criterion)
                avg_acc.append(accuracy)

            avg_acc = sum(avg_acc) / len(avg_acc)
            self._record_text_log("info", f"accuracy: {avg_acc:.4f}")
            self._record_scalar("Eval/accuracy", avg_acc, epoch_count)

    def train_with_val(self, model: nn.Module, data_module: DataModule, loss_module: LossModule, validator) -> None:

        # Datasets & DataLoader
        trainsets = data_module.trainset
        valsets = data_module.valset

        trainsets = ConcatDataset(trainsets.values())
        valsets = ConcatDataset(valsets.values())

        train_loader = data_module.train_loader(trainsets, ddp=self.ngpu > 1)
        val_loader = data_module.val_loader(valsets, ddp=self.ngpu > 1)

        # Model
        model = DDP(model, device_ids=[self.local_rank])
        model.train()

        # Optimizer & Losses
        optimizer = optim.Adam(model.parameters(), lr=self.init_lr)
        criterion = loss_module.__getattribute__(self.loss_type)
        val_criterion = validator

        # AMP
        scaler = GradScaler()

        # Record metrics
        top_models = []
        avg_loss = []
        avg_acc = []

        for epoch_count in range(self.num_epochs):
            self._record_text_log("info", f"Epoch {epoch_count + 1} / {self.num_epochs}")
            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch [{epoch_count + 1}]",
                colour="green",
                disable=not self.enable_log,
            )
            for train_batch in train_pbar:
                train_batch = self._to_device(train_batch)

                optimizer.zero_grad()
                loss: torch.Tensor = self._train_batch(model, train_batch, criterion)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                avg_loss.append(loss.item())

            avg_loss = sum(avg_loss) / len(avg_loss)
            self._record_text_log("info", f"loss: {avg_loss:.6f}")
            self._record_scalar("Train/loss", avg_loss, epoch_count)

            val_pbar = tqdm(
                val_loader,
                desc=f"Eval",
                colour="yellow",
                disable=not self.enable_log,
            )
            for val_batch in val_pbar:
                val_batch = self._to_device(val_batch)
                accuracy: float = self._val_batch(model, val_batch, val_criterion)
                avg_acc.append(accuracy)

            avg_acc = sum(avg_acc) / len(avg_acc)
            self._record_text_log("info", f"accuracy: {avg_acc:.4f}")
            self._record_scalar("Train/accuracy", avg_acc, epoch_count)

            top_models = self._save_topk(top_models, avg_acc, epoch_count, model, optimizer, scaler)

    @torch.inference_mode()
    def vlm_api_validation_baseline(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        data_module: DataModule,
        validator,
        limit_per_dataset: int = 100,
    ) -> float:
        def _tensor2base64(image: torch.Tensor):
            image: Image.Image = T.to_pil_image(image)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{image_base64}"

        dataset = data_module.testset
        dataloader = data_module.test_loader(dataset, ddp=False)
        adapter = CharsetAdapter("0123456789abcdefghijklmnopqrstuvwxyz")

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        message = [
            {
                "role": "system",
                "content": "You are an excellent scene text image recognizer, "
                "give the texts of images directly without any explanation.",
            }
        ]

        response_answers = []
        ground_truths = []
        avg_accuracy = []
        for name, loader in dataloader.items():
            val_pbar = tqdm(
                loader,
                desc=f"VLM {model_name} validate {name}",
                colour="yellow",
            )
            data_count = 0
            accuracy = []
            for val_batch in val_pbar:
                if data_count >= limit_per_dataset:
                    break

                labels, images = val_batch
                labels = labels[: limit_per_dataset - data_count]
                images = images[: limit_per_dataset - data_count]
                data_count += len(labels)

                for label, image in zip(labels, images):
                    message.append(
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": _tensor2base64(image)}}],
                        }
                    )
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=message,
                        )
                        predict = response.choices[0].message.content
                        label = adapter.convert(label)
                        predict = adapter.convert(predict)
                        response_answers.append(predict)
                        ground_truths.append(label)
                        print(f"{label} \t\t {predict}")
                        accuracy.append(int(label == predict))
                        message.pop()
                    except Exception as e:
                        print(f"{e}")
            accuracy = sum(accuracy) / len(accuracy) if accuracy else 0
            print(f"Dataset: {name} Accuracy: {accuracy}")
            avg_accuracy.append(accuracy)

        with open("llm_inference_union14m.log", "w") as f:
            for gt, pred in zip(ground_truths, response_answers):
                f.write(f"{gt}\t\t{pred}\n")
            f.write(f"Accuracy: {accuracy}\n")
            avg_accuracy = sum(avg_accuracy) / len(avg_accuracy) if avg_accuracy else 0
            f.write(f"Avg Accuracy: {avg_accuracy}\n")
        print(f"Avg Accuracy: {avg_accuracy}")
