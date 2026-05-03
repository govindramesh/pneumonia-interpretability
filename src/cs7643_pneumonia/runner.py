from __future__ import annotations

import csv
import json
from pathlib import Path
import random

import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .artifacts import plot_class_balance, plot_confusion_matrix, plot_roc_pr, save_side_by_side_explanations, save_top_examples
from .config import ExperimentConfig
from .data import create_dataloader, load_manifest, manifest_label_ratio, verify_negative_labels, verify_split_integrity
from .interpretability import GradCAM, explain_single_image, save_explanation_bundle, tensor_to_display_image, upsample_map
from .losses import build_loss
from .metrics import compute_binary_metrics, save_json, save_predictions
from .models import build_model, load_checkpoint, resolve_module, save_checkpoint


def _require_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install dependencies from requirements.txt.") from exc
    return torch


def set_seed(seed: int) -> None:
    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(device_name: str) -> object:
    torch = _require_torch()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_loss_to_device(loss_fn: object, device: object) -> object:
    if hasattr(loss_fn, "to"):
        return loss_fn.to(device)
    if hasattr(loss_fn, "pos_weight"):
        loss_fn.pos_weight = loss_fn.pos_weight.to(device)
    return loss_fn


def _flatten_logits(logits: object) -> object:
    return logits.reshape(-1)


def _prepare_output_dirs(config: ExperimentConfig) -> dict[str, Path]:
    experiment_dir = config.experiment_dir()
    paths = {
        "experiment": experiment_dir,
        "checkpoints": experiment_dir / "checkpoints",
        "metrics": experiment_dir / "metrics",
        "predictions": experiment_dir / "predictions",
        "plots": experiment_dir / "plots",
        "interpretability": experiment_dir / "interpretability",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    config.save_resolved(experiment_dir / "resolved_config.json")
    return paths


def _log(message: str) -> None:
    print(message, flush=True)


def _progress(iterable: object, description: str, total: int | None = None, leave: bool = False) -> object:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=description, total=total, leave=leave, dynamic_ncols=True)


def _extract_batch_rows(batch_rows: dict[str, list]) -> list[dict[str, str]]:
    keys = list(batch_rows.keys())
    size = len(batch_rows[keys[0]]) if keys else 0
    rows: list[dict[str, str]] = []
    for idx in range(size):
        rows.append({key: str(batch_rows[key][idx]) for key in keys})
    return rows


def _save_experiment_summary(rows: list[dict[str, object]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_inference(model: object, dataloader: object, device: object, criterion: object | None = None) -> tuple[float, list[dict[str, str | float]], list[int], list[float]]:
    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    prediction_rows: list[dict[str, str | float]] = []
    labels: list[int] = []
    probabilities: list[float] = []
    iterator = _progress(
        dataloader,
        description=f"Eval [{len(dataloader.dataset)} samples]",
        total=len(dataloader),
        leave=False,
    )
    with torch.no_grad():
        for images, batch_labels, batch_rows in iterator:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            logits = _flatten_logits(model(images))
            batch_loss_value = None
            if criterion is not None:
                batch_loss_value = float(criterion(logits, batch_labels).item())
                total_loss += batch_loss_value * images.shape[0]
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            label_values = batch_labels.detach().cpu().numpy().astype(int).tolist()
            if batch_loss_value is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{batch_loss_value:.4f}")
            row_dicts = _extract_batch_rows(batch_rows)
            for row, label, probability in zip(row_dicts, label_values, probs):
                labels.append(int(label))
                probabilities.append(float(probability))
                prediction_rows.append(
                    {
                        "image_id": row["image_id"],
                        "patient_id": row["patient_id"],
                        "label": int(label),
                        "probability": float(probability),
                        "split": row["split"],
                        "image_path": row["image_path"],
                    }
                )
    denom = max(1, len(dataloader.dataset))
    return total_loss / denom, prediction_rows, labels, probabilities


def train_one_epoch(model: object, dataloader: object, optimizer: object, criterion: object, device: object) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0
    iterator = _progress(
        dataloader,
        description=f"Train [{len(dataloader.dataset)} samples]",
        total=len(dataloader),
        leave=False,
    )
    for batch_index, (images, labels, _) in enumerate(iterator, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _flatten_logits(model(images))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.shape[0]
        total_seen += images.shape[0]
        running_loss = total_loss / max(1, total_seen)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(batch_loss=f"{float(loss.item()):.4f}", running_loss=f"{running_loss:.4f}")
    return total_loss / max(1, len(dataloader.dataset))


def train_experiment(config: ExperimentConfig) -> dict:
    torch = _require_torch()
    set_seed(config.seed)
    device = select_device(config.training.device)
    output_dirs = _prepare_output_dirs(config)

    manifest_rows = load_manifest(config.dataset.manifest_path)
    verify_split_integrity(manifest_rows)
    verify_negative_labels(manifest_rows)
    plot_class_balance(manifest_rows, output_dirs["plots"] / "class_balance.png")

    train_loader = create_dataloader(
        manifest_path=config.dataset.manifest_path,
        split="train",
        image_size=config.dataset.image_size,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        train=True,
        subset_size=config.dataset.train_subset_size,
    )
    val_loader = create_dataloader(
        manifest_path=config.dataset.manifest_path,
        split="val",
        image_size=config.dataset.image_size,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        train=False,
        subset_size=config.dataset.eval_subset_size,
        shuffle=False,
    )
    test_loader = create_dataloader(
        manifest_path=config.dataset.manifest_path,
        split="test",
        image_size=config.dataset.image_size,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        train=False,
        subset_size=config.dataset.eval_subset_size,
        shuffle=False,
    )

    model = build_model(config.model, image_size=config.dataset.image_size)
    model.to(device)

    pos_weight = manifest_label_ratio([row for row in manifest_rows if row["split"] == "train"])
    criterion = _move_loss_to_device(build_loss(config.training.loss_name, pos_weight=pos_weight), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    _log(
        f"Starting training for {config.experiment_name} | "
        f"model={config.model.name} | device={device} | "
        f"epochs={config.training.epochs} | batch_size={config.dataset.batch_size} | "
        f"lr={config.training.learning_rate} | loss={config.training.loss_name}"
    )
    _log(
        f"Dataset sizes | train={len(train_loader.dataset)} | "
        f"val={len(val_loader.dataset)} | test={len(test_loader.dataset)} | "
        f"pos_weight={pos_weight:.4f}"
    )

    history: list[dict[str, float | int]] = []
    best_val_auc = -1.0
    best_epoch = 0
    patience = 0
    best_checkpoint = output_dirs["checkpoints"] / "best.pt"

    for epoch in range(1, config.training.epochs + 1):
        _log(f"[Epoch {epoch}/{config.training.epochs}] running...")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, val_labels, val_probs = run_inference(model, val_loader, device, criterion=criterion)
        val_metrics = compute_binary_metrics(val_labels, val_probs, threshold=None, threshold_metric=config.evaluation.threshold_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
                "val_f1": float(val_metrics["f1"]),
            }
        )

        _log(
            f"[Epoch {epoch}/{config.training.epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_roc_auc={float(val_metrics['roc_auc']):.4f} | "
            f"val_pr_auc={float(val_metrics['pr_auc']):.4f} | "
            f"val_f1={float(val_metrics['f1']):.4f}"
        )

        if float(val_metrics["roc_auc"]) > best_val_auc:
            best_val_auc = float(val_metrics["roc_auc"])
            best_epoch = epoch
            patience = 0
            save_checkpoint(model, config.to_dict(), epoch, val_metrics, best_checkpoint)
            _log(
                f"[Epoch {epoch}/{config.training.epochs}] "
                f"new best checkpoint saved to {best_checkpoint} "
                f"(val_roc_auc={best_val_auc:.4f})"
            )
        else:
            patience += 1
            _log(
                f"[Epoch {epoch}/{config.training.epochs}] "
                f"no improvement | early_stop_patience={patience}/{config.training.early_stopping_patience}"
            )

        if config.training.save_every_epoch:
            save_checkpoint(model, config.to_dict(), epoch, val_metrics, output_dirs["checkpoints"] / f"epoch_{epoch:02d}.pt")
            _log(
                f"[Epoch {epoch}/{config.training.epochs}] "
                f"saved epoch checkpoint to {output_dirs['checkpoints'] / f'epoch_{epoch:02d}.pt'}"
            )

        if patience >= config.training.early_stopping_patience:
            _log(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val_roc_auc={best_val_auc:.4f}."
            )
            break

    with (output_dirs["metrics"] / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (output_dirs["metrics"] / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()) if history else ["epoch", "train_loss", "val_loss", "val_roc_auc", "val_pr_auc", "val_f1"])
        writer.writeheader()
        writer.writerows(history)

    model, _ = load_checkpoint(best_checkpoint, config.model, config.dataset.image_size, device)
    _, val_predictions, val_labels, val_probs = run_inference(model, val_loader, device, criterion=None)
    val_metrics = compute_binary_metrics(val_labels, val_probs, threshold=None, threshold_metric=config.evaluation.threshold_metric)
    threshold = float(val_metrics["threshold"])

    test_loss, test_predictions, test_labels, test_probs = run_inference(model, test_loader, device, criterion=criterion)
    test_metrics = compute_binary_metrics(test_labels, test_probs, threshold=threshold, threshold_metric=config.evaluation.threshold_metric)
    test_metrics["loss"] = test_loss
    test_metrics["best_epoch"] = best_epoch

    if config.evaluation.save_predictions:
        save_predictions(val_predictions, output_dirs["predictions"] / "val_predictions.csv")
        save_predictions(test_predictions, output_dirs["predictions"] / "test_predictions.csv")
        save_top_examples(test_predictions, output_dirs["predictions"])

    if config.evaluation.save_plots:
        plot_roc_pr(test_labels, test_probs, output_dirs["plots"], prefix="test")
        plot_confusion_matrix(test_metrics["confusion_matrix"], output_dirs["plots"] / "test_confusion_matrix.png")

    save_json(test_metrics, output_dirs["metrics"] / "test_metrics.json")
    _log(
        f"Finished training for {config.experiment_name} | "
        f"best_epoch={best_epoch} | "
        f"test_roc_auc={float(test_metrics['roc_auc']):.4f} | "
        f"test_pr_auc={float(test_metrics['pr_auc']):.4f} | "
        f"test_f1={float(test_metrics['f1']):.4f}"
    )
    return test_metrics


def evaluate_experiment(config: ExperimentConfig, checkpoint_path: str | Path | None = None) -> dict:
    checkpoint = Path(checkpoint_path) if checkpoint_path else config.experiment_dir() / "checkpoints" / "best.pt"
    output_dirs = _prepare_output_dirs(config)
    device = select_device(config.training.device)
    model, _ = load_checkpoint(checkpoint, config.model, config.dataset.image_size, device)

    val_loader = create_dataloader(
        manifest_path=config.dataset.manifest_path,
        split="val",
        image_size=config.dataset.image_size,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        train=False,
        subset_size=config.dataset.eval_subset_size,
        shuffle=False,
    )
    test_loader = create_dataloader(
        manifest_path=config.dataset.manifest_path,
        split="test",
        image_size=config.dataset.image_size,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        train=False,
        subset_size=config.dataset.eval_subset_size,
        shuffle=False,
    )

    _, val_predictions, val_labels, val_probs = run_inference(model, val_loader, device, criterion=None)
    val_metrics = compute_binary_metrics(val_labels, val_probs, threshold=None, threshold_metric=config.evaluation.threshold_metric)
    _, test_predictions, test_labels, test_probs = run_inference(model, test_loader, device, criterion=None)
    test_metrics = compute_binary_metrics(test_labels, test_probs, threshold=float(val_metrics["threshold"]), threshold_metric=config.evaluation.threshold_metric)

    if config.evaluation.save_predictions:
        save_predictions(val_predictions, output_dirs["predictions"] / "val_predictions.csv")
        save_predictions(test_predictions, output_dirs["predictions"] / "test_predictions.csv")
        save_top_examples(test_predictions, output_dirs["predictions"])
    if config.evaluation.save_plots:
        plot_roc_pr(test_labels, test_probs, output_dirs["plots"], prefix="test")
        plot_confusion_matrix(test_metrics["confusion_matrix"], output_dirs["plots"] / "test_confusion_matrix.png")
    save_json(test_metrics, output_dirs["metrics"] / "test_metrics.json")
    return test_metrics


def interpret_experiment(config: ExperimentConfig, checkpoint_path: str | Path | None = None) -> list[dict[str, float | str]]:
    checkpoint = Path(checkpoint_path) if checkpoint_path else config.experiment_dir() / "checkpoints" / "best.pt"
    output_dirs = _prepare_output_dirs(config)
    device = select_device(config.training.device)
    model, _ = load_checkpoint(checkpoint, config.model, config.dataset.image_size, device)

    loader = create_dataloader(
        manifest_path=config.dataset.manifest_path,
        split="test",
        image_size=config.dataset.image_size,
        batch_size=1,
        num_workers=0,
        train=False,
        subset_size=config.interpretation.num_examples,
        shuffle=False,
    )

    methods = ["gradcam"]
    if config.model.name.lower() == "dinov2_linear":
        methods = ["dino", "dino_rollout"]
    gradcam = None
    if "gradcam" in methods:
        target_name = config.model.gradcam_target_layer or ("layer4" if config.model.name.lower() == "resnet50" else "features.6")
        gradcam = GradCAM(model, resolve_module(model, target_name))

    reports: list[dict[str, float | str]] = []
    for images, _, batch_rows in loader:
        image = images.to(device)
        row = _extract_batch_rows(batch_rows)[0]
        for method_name in methods:
            report = save_explanation_bundle(
                model=model,
                image_tensor=image,
                row=row,
                output_dir=output_dirs["interpretability"],
                method_name=method_name,
                gradcam=gradcam,
                curve_steps=config.interpretation.curve_steps,
                mask_fraction=config.interpretation.mask_fraction,
            )
            reports.append(report)

    if gradcam is not None:
        gradcam.close()

    save_predictions(reports, output_dirs["interpretability"] / "faithfulness_report.csv")
    return reports


def compare_interpretability(
    primary_config: ExperimentConfig,
    comparison_config: ExperimentConfig,
    primary_checkpoint_path: str | Path | None = None,
    comparison_checkpoint_path: str | Path | None = None,
) -> list[str]:
    primary_checkpoint = Path(primary_checkpoint_path) if primary_checkpoint_path else primary_config.experiment_dir() / "checkpoints" / "best.pt"
    comparison_checkpoint = Path(comparison_checkpoint_path) if comparison_checkpoint_path else comparison_config.experiment_dir() / "checkpoints" / "best.pt"
    output_dir = primary_config.experiment_dir() / "interpretability" / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(primary_config.training.device)
    primary_model, _ = load_checkpoint(primary_checkpoint, primary_config.model, primary_config.dataset.image_size, device)
    comparison_model, _ = load_checkpoint(comparison_checkpoint, comparison_config.model, comparison_config.dataset.image_size, device)

    loader = create_dataloader(
        manifest_path=primary_config.dataset.manifest_path,
        split="test",
        image_size=primary_config.dataset.image_size,
        batch_size=1,
        num_workers=0,
        train=False,
        subset_size=primary_config.interpretation.num_examples,
        shuffle=False,
    )

    primary_method = "dino" if primary_config.model.name.lower() == "dinov2_linear" else "gradcam"
    comparison_method = "dino" if comparison_config.model.name.lower() == "dinov2_linear" else "gradcam"

    primary_gradcam = None
    comparison_gradcam = None
    if primary_method == "gradcam":
        primary_target = primary_config.model.gradcam_target_layer or ("layer4" if primary_config.model.name.lower() == "resnet50" else "features.6")
        primary_gradcam = GradCAM(primary_model, resolve_module(primary_model, primary_target))
    if comparison_method == "gradcam":
        comparison_target = comparison_config.model.gradcam_target_layer or ("layer4" if comparison_config.model.name.lower() == "resnet50" else "features.6")
        comparison_gradcam = GradCAM(comparison_model, resolve_module(comparison_model, comparison_target))

    generated: list[str] = []
    for images, _, batch_rows in loader:
        image = images.to(device)
        row = _extract_batch_rows(batch_rows)[0]
        base_image = tensor_to_display_image(image)
        primary_map = explain_single_image(primary_model, image, primary_method, primary_gradcam)
        comparison_map = explain_single_image(comparison_model, image, comparison_method, comparison_gradcam)
        primary_map = upsample_map(primary_map, base_image.shape[0], base_image.shape[1])
        comparison_map = upsample_map(comparison_map, base_image.shape[0], base_image.shape[1])
        output_path = output_dir / f"{row['image_id']}_comparison.png"
        save_side_by_side_explanations(
            base_image=base_image,
            primary_map=primary_map,
            comparison_map=comparison_map,
            output_path=output_path,
            primary_label=primary_config.model.name,
            comparison_label=comparison_config.model.name,
        )
        generated.append(str(output_path.resolve()))

    if primary_gradcam is not None:
        primary_gradcam.close()
    if comparison_gradcam is not None:
        comparison_gradcam.close()
    return generated


def summarize_results(root_dir: str | Path = "artifacts/experiments") -> list[dict[str, object]]:
    root = Path(root_dir)
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("test_metrics.json")):
        experiment_dir = metrics_path.parent.parent
        experiment_name = experiment_dir.name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        resolved_config_path = experiment_dir / "resolved_config.json"
        config_payload = json.loads(resolved_config_path.read_text(encoding="utf-8")) if resolved_config_path.exists() else {}
        rows.append(
            {
                "experiment_name": experiment_name,
                "model": config_payload.get("model", {}).get("name", ""),
                "loss_name": config_payload.get("training", {}).get("loss_name", ""),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "specificity": metrics.get("specificity"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "threshold": metrics.get("threshold"),
            }
        )
    _save_experiment_summary(rows, root / "summary_results.csv")
    return rows
