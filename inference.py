from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed in this Python environment. Install it first with:\n"
        "python3 -m pip install torch torchvision"
    ) from exc

from src.data import bicubic_to_size, list_image_files, load_tensor01
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore low-resolution grayscale images.")
    parser.add_argument("--input-dir", default="/Users/kishan/Downloads/Data-public/train/NoisyLR")
    parser.add_argument("--output-dir", default="outputs/restored")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--output-height", type=int, default=0)
    parser.add_argument("--output-width", type=int, default=0)
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module | None:
    if not checkpoint_path:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    base_channels = int(checkpoint.get("base_channels", 32))
    model = build_model(base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def save_prediction(array: np.ndarray, output_dir: Path, stem: str, save_png: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    array = np.clip(array.astype(np.float32), 0.0, 1.0)
    np.save(output_dir / f"{stem}.npy", array)
    if save_png:
        png = (array * 255.0).round().astype(np.uint8)
        Image.fromarray(png, mode="L").save(output_dir / f"{stem}.png")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)

    input_files = list_image_files(Path(args.input_dir))
    if not input_files:
        raise FileNotFoundError(f"No supported image files were found in {args.input_dir}")
    if args.limit > 0:
        input_files = input_files[: args.limit]

    output_dir = Path(args.output_dir)
    print(f"Device: {device}")
    if model is None:
        print("No checkpoint provided: writing bicubic baseline outputs.")

    for input_path in input_files:
        lr = load_tensor01(input_path)
        if args.output_height > 0 and args.output_width > 0:
            output_size = (args.output_height, args.output_width)
        else:
            output_size = (lr.shape[-2] * args.scale, lr.shape[-1] * args.scale)

        lr_up = bicubic_to_size(lr, output_size).unsqueeze(0).to(device)
        if model is None:
            pred = lr_up
        else:
            pred = model(lr_up)

        array = pred.squeeze(0).squeeze(0).cpu().numpy()
        save_prediction(array, output_dir, input_path.stem, args.save_png)

    print(f"Saved {len(input_files)} restored file(s) to {output_dir}")


if __name__ == "__main__":
    main()