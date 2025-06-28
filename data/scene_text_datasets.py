import io
import torch
import pickle

from pathlib import Path
from data.lmdb_datasets import _LMDBDataset
from PIL import Image
from typing import Literal

from data.augementations import ImageAugmentation
from data.scene_text_edit import SceneTextEdit


class SceneTextDataset(_LMDBDataset):

    def __init__(
        self,
        root,
        path,
        charset,
        image_size,
        cache_file,
        max_label_length,
        max_image_h,
        max_image_w,
        min_label_length=1,
        min_image_h=1,
        min_image_w=1,
        local_rank=0,
        refresh_cache=False,
        # SceneTextDataset
        color_augment: bool = True,
        geometry_augment: bool = True,
    ):
        super().__init__(
            root,
            path,
            charset,
            image_size,
            cache_file,
            max_label_length,
            max_image_h,
            max_image_w,
            min_label_length,
            min_image_h,
            min_image_w,
            local_rank,
            refresh_cache,
        )

        self.image_transform = ImageAugmentation(
            color=color_augment,
            geometry=geometry_augment,
        )

    def transform(
        self,
        label: str,
        image: Image.Image,
    ) -> tuple[str, torch.Tensor]:
        image = self.to_tensor(image)
        image = self.image_transform(image)
        image = self.resize(image)
        return label, image


class SceneTextEditDataset(SceneTextDataset):

    def __init__(
        self,
        root,
        path,
        charset,
        image_size,
        cache_file,
        max_label_length,
        max_image_h,
        max_image_w,
        min_label_length=1,
        min_image_h=1,
        min_image_w=1,
        local_rank=0,
        refresh_cache=False,
        # SceneTextDataset
        color_augment=True,
        geometry_augment=True,
        # SceneTextEditDataset
        font_dir: Path = None,
        edit_prob: float = 1.0,
        edit_types: list[Literal["removal", "substitution", "insertion"]] = None,
        edit_times: list[int] = None,
    ):
        super().__init__(
            root,
            path,
            charset,
            image_size,
            cache_file,
            max_label_length,
            max_image_h,
            max_image_w,
            min_label_length,
            min_image_h,
            min_image_w,
            local_rank,
            refresh_cache,
            color_augment,
            geometry_augment,
        )

        self.edit_transform = SceneTextEdit(
            p=edit_prob,
            charset=charset,
            max_label_length=max_label_length,
            font_dir=font_dir,
            edit_types=edit_types,
            edit_times=edit_times,
        )
        self.weak_image_transform = ImageAugmentation(
            color=False,
            geometry=True,
        )

    def transform(self, label, image, coords: list[list[int]]):
        image = self.to_tensor(image)
        edit_times, (label, image), (edit_label, edit_image) = self.edit_transform(label, image, coords)
        image = self.image_transform(image)
        edit_image = self.weak_image_transform(edit_image)
        image = self.resize(image)
        edit_image = self.resize(edit_image)
        return edit_times, (label, image), (edit_label, edit_image)

    def __getitem__(self, index: int):
        label = self.labels[index]
        index = self.indices[index]

        image_key = f"image-{index:09d}".encode()
        coords_key = f"coordinates-{index:09d}".encode()
        with self.env.begin() as txn:
            image_bin = txn.get(image_key)
            coords = txn.get(coords_key)
        buf = io.BytesIO(image_bin)
        image = Image.open(buf).convert("RGB")

        coords = pickle.loads(coords)
        coords = [[int(c) for c in cs.split(",")] for cs in coords]

        return self.transform(label, image, coords)
