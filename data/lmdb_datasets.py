import io
import lmdb
import unicodedata

from tqdm import tqdm
from PIL import Image
from typing import Iterable, Optional, Union
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from utils.charset_adapter import CharsetAdapter
from utils.loggers import get_text_logger


class _LMDBDataset(Dataset):

    def __init__(
        self,
        root: Union[Path, str],
        path: Union[Path, str],
        charset: str,
        image_size: Iterable[int],
        cache_file: Union[Path, str],
        max_label_length: int,
        max_image_h: int,
        max_image_w: int,
        min_label_length: int = 1,
        min_image_h: int = 1,
        min_image_w: int = 1,
        local_rank: int = 0,
        refresh_cache: bool = False,
    ):
        super().__init__()

        self._env = None
        self.root = Path(root) if isinstance(root, str) else root
        self.path: Path = Path(path) if isinstance(path, str) else path
        self.name = str(path).replace("/", "-")
        self.charset: str = charset
        self.image_size: Iterable[int] = image_size
        self.cache_file: Path = Path(cache_file) if isinstance(cache_file, str) else cache_file
        self.print_enable = local_rank == 0
        self.cache_file = self.cache_file / f"{self.name}_cache.txt"

        self.logger = get_text_logger(__name__)
        self.charset_adapter = CharsetAdapter(charset)
        self.num_samples, self.indices, self.labels = self.filter_data(
            refresh_cache,
            max_label_length,
            max_image_h,
            max_image_w,
            min_label_length,
            min_image_h,
            min_image_w,
        )

        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(image_size)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def print_log(self, method: str, msg: str):
        if self.print_enable:
            self.logger.__getattribute__(method)(msg)

    def _create_env(self):
        return lmdb.Environment(
            path=str(self.root / self.path),
            max_readers=1,
            readonly=True,
            create=False,
            readahead=False,
            meminit=False,
            lock=False,
        )

    def _filter_label(
        self,
        index: int,
        txn: lmdb.Transaction,
        remove_space: bool,
        to_unicode: bool,
        min_length: int,
        max_length: int,
    ) -> Optional[str]:
        label_key = f"label-{index:09d}".encode()
        label: str = txn.get(label_key).decode()
        label = "".join(label.split()) if remove_space else label
        if to_unicode:
            label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
        if not (min_length <= len(label) <= max_length):
            return None
        label = self.charset_adapter.convert(label)
        return label

    def _filter_image(
        self,
        index: int,
        txn: lmdb.Transaction,
        min_h: int,
        max_h: int,
        min_w: int,
        max_w: int,
    ) -> Optional[int]:
        image_key = f"image-{index:09d}".encode()
        buf = io.BytesIO(txn.get(image_key))
        w, h = Image.open(buf).size
        if not (min_h <= h <= max_h and min_w <= w <= max_w):
            return None
        return index

    def preprocess(
        self,
        max_label_length: int,
        max_image_h: int,
        max_image_w: int,
        min_label_length: int = 1,
        min_image_h: int = 1,
        min_image_w: int = 1,
    ) -> tuple[int, list[int], list[str]]:
        remove_space = True
        to_unicode = True
        labels = []
        indices = []

        with self._create_env().begin() as txn:
            num_samples = int(txn.get("num-samples".encode()))
            for index in tqdm(
                range(1, num_samples + 1),
                desc=f"Preprocess {self.path.name}",
                colour="green",
                disable=not self.print_enable,
            ):
                label = self._filter_label(index, txn, remove_space, to_unicode, min_label_length, max_label_length)
                if not label:
                    continue
                index = self._filter_image(index, txn, min_image_h, max_image_h, min_image_w, max_image_w)
                if not index:
                    continue
                labels.append(label)
                indices.append(index)
            self.print_log("info", f"Preprocess finished {len(indices)}/{num_samples}")
        return len(indices), indices, labels

    def store_to_cache(self, indices: list[int], labels: list[str]) -> int:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as cache:
            for i, label in enumerate(labels):
                index = indices[i]
                cache.write(f"{index}\t{label}")
        self.print_log("info", f"Cache stored to {self.cache_file}")
        return len(indices)

    def load_from_cache(self) -> tuple[int, list[int], list[str]]:
        if not self.cache_file.exists():
            self.print_log("error", f"Cache file not exist, please preprocess and save to cache first")
            raise FileNotFoundError(f"Cache file not exist, please preprocess and save to cache first")
        labels = []
        indices = []
        with open(self.cache_file, "r") as cache:
            lines = cache.readlines()
        num_samples = len(lines)
        for line in tqdm(
            lines,
            desc=f"Load from cache {self.name}",
            colour="green",
            disable=not self.print_enable,
        ):
            contents = line.split()
            index = int(contents[0])
            label = contents[1]
            labels.append(label)
            indices.append(index)
        self.print_log("info", f"Load cache finished {len(indices)}/{num_samples}")
        return len(indices), indices, labels

    def filter_data(
        self,
        refresh_cache: bool,
        max_label_length: int,
        max_image_h: int,
        max_image_w: int,
        min_label_length: int,
        min_image_h: int,
        min_image_w: int,
    ) -> tuple[int, list[int], list[str]]:
        if refresh_cache or not self.cache_file.exists():
            self.preprocess(
                max_label_length,
                max_image_h,
                max_image_w,
                min_label_length,
                min_image_h,
                min_image_w,
            )
            self.store_to_cache()
        return self.load_from_cache()

    def transform(self, label: str, image: Image.Image):
        image = self.to_tensor(image)
        image = self.resize(image)
        return label, image

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        label = self.labels[index]
        index = self.indices[index]

        image_key = f"image-{index:09d}".encode()
        with self.env.begin() as txn:
            image_bin = txn.get(image_key)
        buf = io.BytesIO(image_bin)
        image = Image.open(buf).convert("RGB")
        return self.transform(label, image)
