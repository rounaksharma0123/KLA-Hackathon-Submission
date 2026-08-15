from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImagePair:
    lr_path: Path
    gt_path: Path

    @property
    def name(self) -> str:
        return self.lr_path.stem


def list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def pair_by_stem(lr_dir: Path, gt_dir: Path) -> list[ImagePair]:
    lr_files = list_image_files(lr_dir)
    gt_files = list_image_files(gt_dir)
    gt_by_stem = {p.stem: p for p in gt_files}
    return [ImagePair(lr_path=p, gt_path=gt_by_stem[p.stem]) for p in lr_files if p.stem in gt_by_stem]


def find_split_dirs(data_root: Path, split: str) -> tuple[Path, Path]:
    candidates = [
        (data_root / split / "NoisyLR", data_root / split / "GT"),
        (data_root / split / "noisy", data_root / split / "gt"),
        (data_root / split / "LR", data_root / split / "HR"),
    ]
    if split == "train":
        candidates.extend(
            [
                (data_root / "NoisyLR", data_root / "GT"),
                (data_root / "noisy", data_root / "gt"),
                (data_root / "LR", data_root / "HR"),
            ]
        )
    for lr_dir, gt_dir in candidates:
        if lr_dir.exists() or gt_dir.exists():
            return lr_dir, gt_dir
    return candidates[0]


def make_train_val_pairs(data_root: str | Path, val_fraction: float, seed: int) -> tuple[list[ImagePair], list[ImagePair]]:
    root = Path(data_root)
    train_lr_dir, train_gt_dir = find_split_dirs(root, "train")
    train_pairs = pair_by_stem(train_lr_dir, train_gt_dir)

    val_pairs: list[ImagePair] = []
    for split_name in ("val", "validation"):
        val_lr_dir, val_gt_dir = find_split_dirs(root, split_name)
        val_pairs = pair_by_stem(val_lr_dir, val_gt_dir)
        if val_pairs:
            break

    if not train_pairs:
        found_lr = len(list_image_files(train_lr_dir))
        found_gt = len(list_image_files(train_gt_dir))
        public_lr = len(list_image_files(root / "NoisyLR"))
        raise FileNotFoundError(
            "No paired training files were found. Expected matching filenames in "
            f"{train_lr_dir} and {train_gt_dir}. Found {found_lr} low-res training files, "
            f"{found_gt} GT files, and {public_lr} unpaired public low-res files in {root / 'NoisyLR'}."
        )

    if val_pairs:
        return train_pairs, val_pairs

    if len(train_pairs) < 2:
        return train_pairs, train_pairs

    rng = random.Random(seed)
    shuffled = train_pairs[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def load_grayscale_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        array = np.asarray(Image.open(path).convert("L"))

    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        array = array[..., :3].mean(axis=-1)
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = array[:3].mean(axis=0)
    if array.ndim != 2:
        raise ValueError(f"{path} must be a grayscale image or 2D array, got shape {array.shape}")
    return array


def array_to_tensor01(array: np.ndarray) -> torch.Tensor:
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        array = array.astype(np.float32) / float(info.max)
    else:
        array = array.astype(np.float32)
        if np.nanmax(array) > 4.0 or np.nanmin(array) < -1.0:
            array = array / 255.0

    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    return torch.from_numpy(array).unsqueeze(0)


def load_tensor01(path: str | Path) -> torch.Tensor:
    return array_to_tensor01(load_grayscale_array(path))


def bicubic_to_size(tensor: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == tuple(size_hw):
        return tensor
    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=size_hw,
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    return resized.clamp(0.0, 1.0)


def augment_pair(lr: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[2])
        gt = torch.flip(gt, dims=[2])
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[1])
        gt = torch.flip(gt, dims=[1])
    turns = random.randint(0, 3)
    if turns:
        lr = torch.rot90(lr, turns, dims=[1, 2])
        gt = torch.rot90(gt, turns, dims=[1, 2])
    return lr, gt


class PairedRestorationDataset(Dataset):
    def __init__(self, pairs: list[ImagePair], augment: bool = False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index]
        lr = load_tensor01(pair.lr_path)
        gt = load_tensor01(pair.gt_path)
        if self.augment:
            lr, gt = augment_pair(lr, gt)
        lr_up = bicubic_to_size(lr, gt.shape[-2:])
        return lr_up, gt, pair.name
