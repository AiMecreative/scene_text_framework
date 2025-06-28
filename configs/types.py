from typing import Literal, Iterable
from dataclasses import dataclass


@dataclass
class _SceneTextExtraConfigs:

    color_augment: bool
    geometry_augment: bool


@dataclass
class _SceneTextEditExtraConfigs:

    font_dir: str
    edit_prob: float
    edit_types: list[Literal["removal", "substitution", "insertion"]]
    edit_times: list[int]


@dataclass
class DataLoaderConfigs:

    num_workers: int
    batch_size: int


@dataclass
class DataModuleConfigs:

    root: str
    train: list[str]
    val: list[str]
    test: list[str]
    train_charset: str
    test_charset: str
    image_size: Iterable[int]
    cache_file: str
    max_label_length: int
    max_image_h: int
    max_image_w: int
    min_label_length: int
    min_image_h: int
    min_image_w: int
    refresh_cache: bool

    scene_text_extra: _SceneTextExtraConfigs
    scene_text_edit_extra: _SceneTextEditExtraConfigs
    dataloader: DataLoaderConfigs


@dataclass
class TrainerConfigs:

    ngpu: int
    num_epochs: int
    init_lr: float
    loss_type: Literal["model_loss", "contrastive_loss"]
    save_dir: str
    log_dir: str
