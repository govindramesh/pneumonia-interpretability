from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import summarize_manifest
from .metrics import pr_curve, roc_curve


def _require_plotting() -> object:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cs7643-matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for report artifact generation.") from exc
    return plt


def plot_class_balance(rows: list[dict[str, str]], output_path: str | Path) -> None:
    plt = _require_plotting()
    summary = summarize_manifest(rows)
    splits = ["train", "val", "test"]
    positives = [summary.get(split, {}).get("positive", 0) for split in splits]
    negatives = [summary.get(split, {}).get("negative", 0) for split in splits]
    x = np.arange(len(splits))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.15, negatives, width=0.3, label="No Finding")
    ax.bar(x + 0.15, positives, width=0.3, label="Pneumonia")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_title("Class Balance by Split")
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_pr(labels: list[int], probabilities: list[float], output_dir: str | Path, prefix: str) -> None:
    plt = _require_plotting()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_array = np.asarray(labels, dtype=np.int32)
    probability_array = np.asarray(probabilities, dtype=np.float64)

    roc_x, roc_y = roc_curve(label_array, probability_array)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(roc_x, roc_y, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=200)
    plt.close(fig)

    pr_x, pr_y = pr_curve(label_array, probability_array)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pr_x, pr_y, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_pr.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(confusion: dict[str, int], output_path: str | Path) -> None:
    plt = _require_plotting()
    matrix = np.array(
        [
            [confusion["tn"], confusion["fp"]],
            [confusion["fn"], confusion["tp"]],
        ],
        dtype=np.int32,
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_top_examples(predictions: list[dict[str, str | float]], output_dir: str | Path, top_k: int = 10) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(predictions, key=lambda row: abs(float(row["probability"]) - float(row["label"])), reverse=True)
    worst = ranked[:top_k]
    best = list(reversed(ranked[-top_k:]))
    for name, rows in (("top_errors.csv", worst), ("top_correct.csv", best)):
        path = output_dir / name
        fieldnames = list(rows[0].keys()) if rows else ["image_id", "patient_id", "label", "probability", "split"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def save_overlay_grid(base_image: np.ndarray, saliency_map: np.ndarray, output_path: str | Path, title: str) -> None:
    plt = _require_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(base_image, cmap="gray" if base_image.ndim == 2 else None)
    axes[0].set_title("Image")
    axes[1].imshow(saliency_map, cmap="inferno")
    axes[1].set_title("Saliency")
    axes[2].imshow(base_image, cmap="gray" if base_image.ndim == 2 else None)
    axes[2].imshow(saliency_map, cmap="inferno", alpha=0.4)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_faithfulness_curve(fractions: list[float], scores: list[float], output_path: str | Path, title: str) -> None:
    plt = _require_plotting()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fractions, scores, marker="o")
    ax.set_xlabel("Masked Fraction")
    ax.set_ylabel("Predicted Pneumonia Probability")
    ax.set_title(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_side_by_side_explanations(base_image: np.ndarray, primary_map: np.ndarray, comparison_map: np.ndarray, output_path: str | Path, primary_label: str, comparison_label: str) -> None:
    plt = _require_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for row_idx, (saliency, label) in enumerate(((primary_map, primary_label), (comparison_map, comparison_label))):
        axes[row_idx, 0].imshow(base_image, cmap="gray" if base_image.ndim == 2 else None)
        axes[row_idx, 0].set_title("Image")
        axes[row_idx, 1].imshow(saliency, cmap="inferno")
        axes[row_idx, 1].set_title(f"{label} Map")
        axes[row_idx, 2].imshow(base_image, cmap="gray" if base_image.ndim == 2 else None)
        axes[row_idx, 2].imshow(saliency, cmap="inferno", alpha=0.4)
        axes[row_idx, 2].set_title(f"{label} Overlay")
    for axis in axes.reshape(-1):
        axis.axis("off")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
