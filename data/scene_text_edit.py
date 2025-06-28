import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as trans_fn

from copy import deepcopy
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path
from typing import Literal, Union


class SceneTextEdit:

    def __init__(
        self,
        p: float,
        charset: str,
        max_label_length: int,
        font_dir: Path,
        edit_types: list[Literal["removal", "substitution", "insertion"]],
        edit_times: list[int],
    ):
        self._patches = self._create_char_patch_dict(font_dir, charset, 36)
        self.edit_types = edit_types
        self.edit_times = edit_times
        self.max_label_length = max_label_length
        self.p = p

        self.op_map = {
            "removal": self._removal,
            "substitution": self._substitution,
            "insertion": self._insertion,
        }

    def _create_char_patch_dict(
        self,
        font_dir: Union[Path, str],
        charset: str,
        size: int,
    ) -> dict[str, torch.Tensor]:
        patch_dict = {}
        font_dir = Path(font_dir) if isinstance(font_dir, str) else font_dir
        fonts = os.listdir(font_dir)
        idx = random.randint(0, len(fonts))
        font = ImageFont.truetype(font_dir / fonts[idx], size=size)
        for i, char in enumerate(charset):
            img = Image.new("RGB", (size, size), color="black")
            draw = ImageDraw.Draw(img)
            _, _, w, h = draw.textbbox((0, 0), text=char, font=font)
            draw.text(((size - w) // 2, (size - h) // 2), char, font=font, fill="white", stroke_width=1)
            patch = trans_fn.to_tensor(img)
            # Resize the patch to make the character filled the img
            gray = patch[0]
            coords = (gray > 0).nonzero(as_tuple=False)
            if coords.numel() > 0:
                y_min, x_min = coords.min(dim=0).values
                y_max, x_max = coords.max(dim=0).values + 1

                cropped = patch[:, y_min:y_max, x_min:x_max].unsqueeze(0)  # [1, 3, H', W']
                resized = F.interpolate(cropped, size=(size, size), mode="bilinear", align_corners=False)
                patch = resized.squeeze(0)  # [3, size, size]
            patch_dict[char] = patch
        return patch_dict

    def _removal(
        self,
        label: str,
        image: torch.Tensor,
        pos: int,
        h1: int,
        h2: int,
        w1: int,
        w2: int,
        bg: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        if len(label) > 1:
            image[:, h1:h2, w1:w2] = bg
            label = label[:pos] + label[pos + 1 :]
        return label, image

    def _substitution(
        self,
        label: str,
        image: torch.Tensor,
        pos: int,
        h1: int,
        h2: int,
        w1: int,
        w2: int,
        bg: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        char = random.choice(list(self._patches.keys()))
        patch = self._patches[char]
        while char == label[pos]:
            char = random.choice(list(self._patches.keys()))
            patch = self._patches[char]
        patch = trans_fn.resize(patch, [h2 - h1, w2 - w1])
        bg = torch.clamp(bg + patch, max=1.0)
        image[:, h1:h2, w1:w2] = bg
        label = label[:pos] + char + label[pos + 1 :]
        return label, image

    def _insertion(
        self,
        label: str,
        image: torch.Tensor,
        pos: int,
        h1: int,
        h2: int,
        w1: int,
        w2: int,
        bg: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        if len(label) < self.max_label_length:
            char1 = label[pos]
            patch1 = self._patches[char1]
            char2 = random.choice(list(self._patches.keys()))
            patch2 = self._patches[char2]
            patch1 = trans_fn.resize(patch1, [h2 - h1, (w2 - w1) // 2])
            patch2 = trans_fn.resize(patch2, [h2 - h1, (w2 - w1) - (w2 - w1) // 2])
            patch = torch.concat([patch1, patch2], dim=-1)
            bg = torch.clamp(bg + patch, max=1.0)
            image[:, h1:h2, w1:w2] = bg
            label = label[:pos] + char2 + label[pos + 1 :]
        return label, image

    def __call__(
        self,
        label: str,
        image: torch.Tensor,
        coords: list[list[int]],
    ) -> tuple[int, tuple[str, torch.Tensor], tuple[str, torch.Tensor]]:
        edit_image = deepcopy(image)
        edit_label = deepcopy(label)

        if random.random() > self.p:
            return 0, (label, image), (edit_label, edit_image)

        edit_times = 0
        loops = random.randint(*self.edit_times)
        for _ in range(loops):
            prev_edit_label = edit_label
            length = len(edit_label)
            pos = random.randint(0, length - 1)
            coord = coords[pos]
            min_w = min_h = 0
            max_w = image.shape[-1]
            max_h = image.shape[-2]
            w1 = max(coord[0], min_w)
            h1 = max(coord[1], min_h)
            w2 = min(coord[2], max_w)
            h2 = min(coord[3], max_h)

            bg = image[:, h1:h2, w1:w2].mean(dim=(1, 2), keepdim=True)

            op = self.op_map[random.choice(self.edit_types)]
            edit_label, edit_image = op(edit_label, edit_image, pos, h1, h2, w1, w2, bg)
            edit_times += int(prev_edit_label != edit_label)

        return edit_times, (label, image), (edit_label, edit_image)
