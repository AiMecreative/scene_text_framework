import os
import torch
import torch.nn as nn

from abc import ABC
from typing import Union, Optional
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf
from dataclasses import dataclass

from models.model_adapter import ModelAdapter
from configs.types import ModelModuleConfigs


@dataclass
class CheckpointKeywords:

    MODEL: str = "model"
    OPTIMIZER: str = "optimizer"
    EPOCH: str = "epoch"
    SCALER: str = "scaler"


class ModelModule:

    def __init__(self, configs):

        self.model: nn.Module = instantiate(configs.model)
        self.adapter: ModelAdapter = instantiate(configs.adapter)

        self.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        self.model = self.model.to(self.local_rank)

        self.resume_or_init(configs.checkpoint_path)
        self.ddp = configs.ngpu > 1
        if self.ddp:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)

    def resume_or_init(self, checkpoint_path: Optional[Union[str, Path]]) -> None:
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint[CheckpointKeywords.MODEL])
        else:
            # If the model has a linear layer, we need to initialize it
            for layer in self.model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            # If the model has a convolutional layer, we need to initialize it
            for layer in self.model.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            # If the model has a batch normalization layer, we need to initialize it
            for layer in self.model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
            # If the model has a layer normalization layer, we need to initialize it
            for layer in self.model.modules():
                if isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    @property
    def parameters(self):
        """
        Returns the parameters of the model.
        If the model is wrapped in DDP, it returns the parameters of the underlying model.
        """
        return self.model.module.parameters() if self.ddp else self.model.parameters()

    @property
    def model_state_dict(self):
        """
        Returns the state dict of the model.
        If the model is wrapped in DDP, it returns the state dict of the underlying model.
        """
        return self.model.module.state_dict() if self.ddp else self.model.state_dict()

    def unzip_train_batch(self, batch):
        return self.adapter.train_input(batch, self.local_rank)

    def unzip_val_batch(self, batch):
        return self.adapter.val_input(batch, self.local_rank)

    def train_batch(self, batch):
        return self.adapter.model_train_forward(self.model, batch)

    @torch.inference_mode()
    def val_batch(self, batch):
        return self.adapter.model_inference_forward(self.model, batch)
    
