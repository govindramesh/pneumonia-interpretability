from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def ensure_probabilities(values: Iterable[float], already_probabilities: bool = True) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if already_probabilities:
        return np.clip(array, 0.0, 1.0)
    return sigmoid(array)


def confusion_from_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    predictions = (probabilities >= threshold).astype(np.int32)
    labels = labels.astype(np.int32)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def precision_recall_f1(confusion: dict[str, int]) -> tuple[float, float, float]:
    precision = safe_divide(confusion["tp"], confusion["tp"] + confusion["fp"])
    recall = safe_divide(confusion["tp"], confusion["tp"] + confusion["fn"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def specificity(confusion: dict[str, int]) -> float:
    return safe_divide(confusion["tn"], confusion["tn"] + confusion["fp"])


def accuracy(confusion: dict[str, int]) -> float:
    total = confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"]
    return safe_divide(confusion["tp"] + confusion["tn"], total)


def roc_curve(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(probabilities)[::-1]
    thresholds = np.concatenate(([1.0], thresholds, [0.0]))
    tprs = []
    fprs = []
    for threshold in thresholds:
        confusion = confusion_from_threshold(labels, probabilities, float(threshold))
        tpr = safe_divide(confusion["tp"], confusion["tp"] + confusion["fn"])
        fpr = safe_divide(confusion["fp"], confusion["fp"] + confusion["tn"])
        tprs.append(tpr)
        fprs.append(fpr)
    order = np.argsort(fprs)
    return np.asarray(fprs)[order], np.asarray(tprs)[order]


def pr_curve(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(probabilities)[::-1]
    precisions = []
    recalls = []
    for threshold in thresholds:
        confusion = confusion_from_threshold(labels, probabilities, float(threshold))
        precision, recall, _ = precision_recall_f1(confusion)
        precisions.append(precision)
        recalls.append(recall)
    precisions = np.asarray([1.0] + precisions + [0.0])
    recalls = np.asarray([0.0] + recalls + [1.0])
    order = np.argsort(recalls)
    return recalls[order], precisions[order]


def auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray, metric_name: str = "f1") -> float:
    thresholds = np.unique(probabilities)
    if len(thresholds) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        confusion = confusion_from_threshold(labels, probabilities, float(threshold))
        precision, recall, f1 = precision_recall_f1(confusion)
        score_map = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "specificity": specificity(confusion),
            "accuracy": accuracy(confusion),
        }
        score = score_map.get(metric_name, f1)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def compute_binary_metrics(labels: Iterable[float], probabilities: Iterable[float], threshold: float | None = None, threshold_metric: str = "f1") -> dict[str, float | dict[str, int]]:
    label_array = np.asarray(list(labels), dtype=np.int32)
    probability_array = ensure_probabilities(probabilities)
    chosen_threshold = choose_threshold(label_array, probability_array, metric_name=threshold_metric) if threshold is None else float(threshold)
    confusion = confusion_from_threshold(label_array, probability_array, chosen_threshold)
    precision, recall, f1 = precision_recall_f1(confusion)
    roc_x, roc_y = roc_curve(label_array, probability_array)
    pr_x, pr_y = pr_curve(label_array, probability_array)
    return {
        "threshold": chosen_threshold,
        "accuracy": accuracy(confusion),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity(confusion),
        "f1": f1,
        "roc_auc": auc(roc_x, roc_y),
        "pr_auc": auc(pr_x, pr_y),
        "confusion_matrix": confusion,
    }


def save_json(payload: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_predictions(rows: list[dict[str, str | float]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["image_id", "patient_id", "label", "probability", "split"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
