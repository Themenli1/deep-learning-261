"""Generate a correct/incorrect prediction grid from a trained checkpoint.

Run this in the training environment (dataset and checkpoint available), e.g.:

    python src/qualitative_examples.py \
        --config configs/pretrained_frozen.yaml \
        --checkpoint checkpoints/resnet50_frozen/resnet50_best.pt \
        --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from datasets.food101 import create_datasets
from models import create_model


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save correct/incorrect Food-101 predictions")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-correct", type=int, default=4)
    parser.add_argument("--num-incorrect", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for example selection")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1,
        help="Maximum examples per true class within a group (1 keeps the grid varied)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )

    root = args.data_root or data_cfg.get("root", "data/food-101")
    train_ds, val_ds, test_ds, classes = create_datasets(
        root=root,
        image_size=data_cfg.get("image_size", 224),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        seed=config.get("seed", 42),
    )
    dataset = val_ds if args.split == "val" else test_ds
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(dataset))
    subset = torch.utils.data.Subset(dataset, order.tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_model(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    correct: list[tuple[np.ndarray, str, str]] = []
    incorrect: list[tuple[np.ndarray, str, str]] = []
    seen_correct: dict[str, int] = {}
    seen_incorrect: dict[str, int] = {}

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()

            for image, target, prediction in zip(images.cpu(), targets, predictions):
                true_name = classes[int(target)]
                pred_name = classes[int(prediction)]
                item = (denormalize(image), true_name, pred_name)
                if int(target) == int(prediction):
                    count = seen_correct.get(true_name, 0)
                    if len(correct) < args.num_correct and count < args.max_per_class:
                        correct.append(item)
                        seen_correct[true_name] = count + 1
                else:
                    count = seen_incorrect.get(true_name, 0)
                    if len(incorrect) < args.num_incorrect and count < args.max_per_class:
                        incorrect.append(item)
                        seen_incorrect[true_name] = count + 1
            if len(correct) >= args.num_correct and len(incorrect) >= args.num_incorrect:
                break

    items = correct + incorrect
    if not items:
        raise SystemExit("No examples collected; check the checkpoint and split.")

    columns = max(len(correct), len(incorrect), 1)
    rows = 2
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 8.0))
    axes = np.atleast_2d(axes)

    for row, (group, title, color) in enumerate([
        (correct, "Correct predictions", "#1b7a2b"),
        (incorrect, "Incorrect predictions", "#b03a2e"),
    ]):
        for col in range(columns):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(group):
                image, true_name, pred_name = group[col]
                ax.imshow(image)
                ax.set_title(
                    f"true: {true_name.replace('_', ' ')}\npred: {pred_name.replace('_', ' ')}",
                    fontsize=9,
                    color=color,
                )
        axes[row, 0].text(
            -0.08, 0.5, title, transform=axes[row, 0].transAxes,
            rotation=90, va="center", ha="right", fontsize=11, fontweight="bold", color=color,
        )

    fig.suptitle(f"Food-101 qualitative examples ({args.split} split)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / Path(args.checkpoint).parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qualitative_examples.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    docs_path = Path(__file__).resolve().parents[2] / "docs" / "assets" / "a2" / "qualitative_examples.png"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_path, docs_path)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "correct_examples": len(correct),
        "incorrect_examples": len(incorrect),
        "output": str(output_path),
    }
    with open(output_dir / "qualitative_examples.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
