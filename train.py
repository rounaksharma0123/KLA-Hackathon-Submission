from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed in this Python environment. Install it first with:\n"
        "python3 -m pip install torch torchvision"
    ) from exc

from src.data import PairedRestorationDataset, make_train_val_pairs
from src.model import build_model


def load_simple_yaml(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}

    values: dict[str, object] = {}
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if not value:
            continue
        if value.lower() in {"true", "false"}:
            values[key] = value.lower() == "true"
        else:
            try:
                values[key] = int(value)
            except ValueError:
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = value.strip("'\"")
    return values


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/default.yaml")
    config_args, _ = config_parser.parse_known_args()
    defaults = load_simple_yaml(config_args.config)

    parser = argparse.ArgumentParser(description="Train a lightweight grayscale image restoration model.")
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--data-root", default=defaults.get("data_root", "/Users/kishan/Downloads/Data-public"))
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 50)))
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 4)))
    parser.add_argument("--lr", type=float, default=float(defaults.get("learning_rate", 0.0002)))
    parser.add_argument("--base-channels", type=int, default=int(defaults.get("base_channels", 32)))
    parser.add_argument("--val-fraction", type=float, default=float(defaults.get("val_fraction", 0.1)))
    parser.add_argument("--num-workers", type=int, default=int(defaults.get("num_workers", 0)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--device", default=str(defaults.get("device", "auto")))
    parser.add_argument("--output", default=str(defaults.get("output", "checkpoints/best_model.pt")))
    parser.add_argument("--overfit-count", type=int, default=int(defaults.get("overfit_count", 0)))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    total_squared_error = 0.0

    for lr_up, gt, _ in loader:
        lr_up = lr_up.to(device)
        gt = gt.to(device)
        pred = model(lr_up)
        loss = criterion(pred, gt)
        total_loss += loss.item() * lr_up.size(0)
        total_squared_error += torch.sum((pred - gt) ** 2).item()
        total_pixels += gt.numel()

    mean_loss = total_loss / max(1, len(loader.dataset))
    mean_mse = total_squared_error / max(1, total_pixels)
    return mean_loss, psnr_from_mse(mean_mse)


@torch.no_grad()
def evaluate_bicubic_baseline(loader: DataLoader, device: torch.device) -> float:
    total_pixels = 0
    total_squared_error = 0.0
    for lr_up, gt, _ in loader:
        lr_up = lr_up.to(device)
        gt = gt.to(device)
        total_squared_error += torch.sum((lr_up - gt) ** 2).item()
        total_pixels += gt.numel()
    return psnr_from_mse(total_squared_error / max(1, total_pixels))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    try:
        train_pairs, val_pairs = make_train_val_pairs(args.data_root, args.val_fraction, args.seed)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if args.overfit_count > 0:
        train_pairs = train_pairs[: args.overfit_count]
        val_pairs = train_pairs
        print(f"Overfit mode: using the same {len(train_pairs)} image(s) for train and validation.")

    train_dataset = PairedRestorationDataset(train_pairs, augment=args.overfit_count == 0)
    val_dataset = PairedRestorationDataset(val_pairs, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(base_channels=args.base_channels).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Train pairs: {len(train_dataset)} | Validation pairs: {len(val_dataset)}")
    print(f"Bicubic validation PSNR before training: {evaluate_bicubic_baseline(val_loader, device):.2f} dB")

    best_psnr = -float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for lr_up, gt, _ in train_loader:
            lr_up = lr_up.to(device)
            gt = gt.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lr_up)
            loss = criterion(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * lr_up.size(0)

        train_loss = running_loss / max(1, len(train_dataset))
        val_loss, val_psnr = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train L1 {train_loss:.5f} | val L1 {val_loss:.5f} | val PSNR {val_psnr:.2f} dB"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "base_channels": args.base_channels,
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "data_root": str(args.data_root),
                },
                output_path,
            )
            print(f"Saved new best checkpoint to {output_path}")

    print(f"Done. Best validation PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
