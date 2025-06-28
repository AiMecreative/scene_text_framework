import lmdb
import torch

from dataclasses import asdict
from pathlib import Path
from typing import Union, Iterable, Any
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from configs.types import DataModuleConfigs
from data.scene_text_datasets import SceneTextDataset, SceneTextEditDataset


class DataModule:

    def __init__(self, configs: DataModuleConfigs):

        self.train = [Path(f) for f in configs.train]
        self.val = [Path(f) for f in configs.val]
        self.test = [Path(f) for f in configs.test]
        self.train_charset = configs.train_charset
        self.test_charset = configs.test_charset

        self.base_cfgs = asdict(configs)
        self.st_cfgs = asdict(configs.scene_text_extra)
        self.edit_cfgs = asdict(configs.scene_text_edit_extra)
        self.loader_cfgs = asdict(configs.dataloader)

        for k in [
            "train",
            "val",
            "test",
            "dataloader",
            "train_charset",
            "test_charset",
            "scene_text_extra",
            "scene_text_edit_extra",
        ]:
            self.base_cfgs.pop(k)

    def _parse_dataset_name(self, path: Union[Path, str]) -> str:
        if isinstance(path, Path):
            path = str(path)
        path = path.replace("/", "-")
        return path

    @property
    def trainset(self) -> dict[str, Dataset]:
        argdict: dict[str, Any] = self.base_cfgs
        argdict.update(self.st_cfgs)
        traindict = {
            self._parse_dataset_name(path): SceneTextDataset(
                path=path,
                charset=self.train_charset,
                **argdict,
            )
            for path in self.train
        }
        return traindict

    @property
    def traineditset(self) -> dict[str, Dataset]:
        argdict: dict[str, Any] = self.base_cfgs
        argdict.update(self.edit_cfgs)
        traindict = {
            self._parse_dataset_name(path): SceneTextEditDataset(
                path=path,
                charset=self.train_charset,
                **argdict,
            )
            for path in self.train
        }
        return traindict

    @property
    def testset(self) -> dict[str, Dataset]:
        argdict: dict[str, Any] = self.base_cfgs
        testdict = {
            self._parse_dataset_name(path): SceneTextDataset(
                path=path,
                charset=self.test_charset,
                color_augment=False,
                geometry_augment=False,
                **argdict,
            )
            for path in self.test
        }
        return testdict

    @property
    def valset(self) -> dict[str, Dataset]:
        argdict: dict[str, Any] = self.base_cfgs
        valdict = {
            self._parse_dataset_name(path): SceneTextDataset(
                path=path,
                charset=self.test_charset,
                color_augment=False,
                geometry_augment=False,
                **argdict,
            )
            for path in self.val
        }
        return valdict

    def train_loader(self, dataset: Dataset, ddp: bool = False) -> DataLoader:
        sampler = DistributedSampler(dataset) if ddp else None
        return DataLoader(
            dataset=dataset,
            shuffle=sampler is None,
            sampler=sampler,
            **self.loader_cfgs,
        )

    def trainedit_loader(self, dataset: Dataset, ddp: bool = False) -> DataLoader:
        return self.train_loader(dataset, ddp)

    def val_loader(self, dataset: Dataset, ddp: bool = False) -> DataLoader:
        sampler = DistributedSampler(dataset) if ddp else None
        return DataLoader(
            dataset=dataset,
            shuffle=False,
            sampler=sampler,
            **self.loader_cfgs,
        )

    def test_loader(self, datasets: dict[str, Dataset], ddp: bool = False) -> dict[str, DataLoader]:
        loaders = {}
        for name, dataset in datasets.items():
            sampler = DistributedSampler(dataset) if ddp else None
            loaders[name] = DataLoader(
                dataset=dataset,
                shuffle=True,
                sampler=sampler,
                **self.loader_cfgs,
            )
        return loaders
