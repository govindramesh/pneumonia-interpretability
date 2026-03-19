from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np


@dataclass
class ManifestRow:
    image_id: str
    patient_id: str
    label: int
    split: str
    image_path: str
    finding_labels: str


def _require_torchvision() -> tuple[object, object]:
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install dependencies from requirements.txt.") from exc
    return Dataset, DataLoader


def _require_image_stack() -> tuple[object, object]:
    try:
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Pillow and torchvision are required. Install dependencies from requirements.txt.") from exc
    return Image, transforms


def _require_hf_datasets() -> object:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset support requires `datasets` and `huggingface_hub`. "
            "Install dependencies from requirements.txt."
        ) from exc
    return load_dataset


def load_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_manifest(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        split = row["split"]
        if split not in summary:
            summary[split] = {"total": 0, "positive": 0, "negative": 0, "patients": 0}
        summary[split]["total"] += 1
        if int(row["label"]) == 1:
            summary[split]["positive"] += 1
        else:
            summary[split]["negative"] += 1

    patients_by_split: dict[str, set[str]] = {}
    for row in rows:
        patients_by_split.setdefault(row["split"], set()).add(row["patient_id"])
    for split, patients in patients_by_split.items():
        summary.setdefault(split, {})["patients"] = len(patients)
    return summary


def find_image_paths(image_root: str | Path) -> dict[str, str]:
    root = Path(image_root)
    mapping: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            mapping[path.name] = str(path.resolve())
    return mapping


def filter_chestxray14_row(row: dict[str, str]) -> tuple[int | None, str]:
    labels_raw = row.get("Finding Labels", "")
    labels = [item.strip() for item in labels_raw.split("|") if item.strip()]
    if "Pneumonia" in labels:
        return 1, labels_raw
    if labels == ["No Finding"]:
        return 0, labels_raw
    return None, labels_raw


def filter_hf_labels(labels_raw: object) -> tuple[int | None, str]:
    if isinstance(labels_raw, str):
        labels = [item.strip() for item in labels_raw.split("|") if item.strip()]
    else:
        labels = [str(item).strip() for item in labels_raw or [] if str(item).strip()]
    if "Pneumonia" in labels:
        return 1, "|".join(labels)
    if labels == ["No Finding"]:
        return 0, "No Finding"
    return None, "|".join(labels)


def _patient_label_groups(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    patient_has_positive: dict[str, bool] = {}
    for row in rows:
        patient_id = row["patient_id"]
        patient_has_positive[patient_id] = patient_has_positive.get(patient_id, False) or int(row["label"]) == 1
    positive = sorted([patient for patient, is_positive in patient_has_positive.items() if is_positive])
    negative = sorted([patient for patient, is_positive in patient_has_positive.items() if not is_positive])
    return positive, negative


def _split_patients(patients: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    items = list(patients)
    rng.shuffle(items)
    total = len(items)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    train = set(items[:n_train])
    val = set(items[n_train:n_train + n_val])
    test = set(items[n_train + n_val:])
    if total > 0 and not test:
        moved = val.pop() if val else train.pop()
        test.add(moved)
    return {"train": train, "val": val, "test": test}


def build_patient_level_splits(rows: list[dict[str, str]], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> list[dict[str, str]]:
    positive_patients, negative_patients = _patient_label_groups(rows)
    pos_splits = _split_patients(positive_patients, ratios, seed)
    neg_splits = _split_patients(negative_patients, ratios, seed + 1)
    patient_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for patient_id in pos_splits[split] | neg_splits[split]:
            patient_to_split[patient_id] = split
    updated: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        copied["split"] = patient_to_split[row["patient_id"]]
        updated.append(copied)
    return updated


def write_manifest(rows: list[dict[str, str]], output_manifest: str | Path) -> None:
    path = Path(output_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "patient_id", "label", "split", "image_path", "finding_labels"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_chestxray14_manifest(metadata_csv: str | Path, image_root: str | Path, output_manifest: str | Path, seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, dict[str, int]]:
    image_map = find_image_paths(image_root)
    filtered_rows: list[dict[str, str]] = []
    with Path(metadata_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label, finding_labels = filter_chestxray14_row(row)
            if label is None:
                continue
            image_id = row["Image Index"]
            if image_id not in image_map:
                continue
            patient_id = row["Patient ID"]
            filtered_rows.append(
                {
                    "image_id": image_id,
                    "patient_id": patient_id,
                    "label": str(label),
                    "split": "",
                    "image_path": image_map[image_id],
                    "finding_labels": finding_labels,
                }
            )

    split_rows = build_patient_level_splits(filtered_rows, seed=seed, ratios=ratios)
    write_manifest(split_rows, output_manifest)
    return summarize_manifest(split_rows)


def prepare_hf_chestxray14_manifest(
    hf_dataset: str,
    output_manifest: str | Path,
    output_image_dir: str | Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    hf_splits: tuple[str, ...] = ("train", "valid", "test"),
    streaming: bool = True,
    limit_per_class: int | None = None,
) -> dict[str, dict[str, int]]:
    load_dataset = _require_hf_datasets()
    Image, _ = _require_image_stack()

    output_image_dir = Path(output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    filtered_rows: list[dict[str, str]] = []
    saved_counts = {0: 0, 1: 0}

    for split_name in hf_splits:
        dataset = load_dataset(hf_dataset, split=split_name, streaming=streaming)
        for index, sample in enumerate(dataset):
            label, finding_labels = filter_hf_labels(sample.get("label"))
            if label is None:
                continue
            if limit_per_class is not None and saved_counts[label] >= limit_per_class:
                continue

            patient_id = str(sample.get("Patient ID"))
            image = sample.get("image")
            if image is None:
                continue

            image_id = f"hf_{split_name}_{index:06d}_patient_{patient_id}.png"
            image_path = output_image_dir / image_id

            if not image_path.exists():
                if isinstance(image, Image.Image):
                    pil_image = image
                else:
                    # `datasets` image feature usually decodes to PIL already, but keep a fallback.
                    pil_image = Image.fromarray(np.asarray(image))
                pil_image.convert("L").save(image_path)

            filtered_rows.append(
                {
                    "image_id": image_id,
                    "patient_id": patient_id,
                    "label": str(label),
                    "split": "",
                    "image_path": str(image_path.resolve()),
                    "finding_labels": finding_labels,
                }
            )
            saved_counts[label] += 1

    split_rows = build_patient_level_splits(filtered_rows, seed=seed, ratios=ratios)
    write_manifest(split_rows, output_manifest)
    return summarize_manifest(split_rows)


def prepare_kaggle_pneumonia_manifest(
    dataset_root: str | Path,
    output_manifest: str | Path,
    split_dirs: tuple[str, ...] = ("train", "val", "test"),
    normal_dir_name: str = "NORMAL",
    pneumonia_dir_name: str = "PNEUMONIA",
) -> dict[str, dict[str, int]]:
    root = Path(dataset_root)
    rows: list[dict[str, str]] = []

    split_aliases = {
        "train": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }

    for split_dir in split_dirs:
        normalized_split = split_aliases.get(split_dir.lower(), split_dir.lower())
        for class_dir_name, label, finding_labels in (
            (normal_dir_name, 0, "No Finding"),
            (pneumonia_dir_name, 1, "Pneumonia"),
        ):
            class_dir = root / split_dir / class_dir_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                image_id = image_path.name
                synthetic_patient_id = f"{normalized_split}_{class_dir_name.lower()}_{image_path.stem}"
                rows.append(
                    {
                        "image_id": image_id,
                        "patient_id": synthetic_patient_id,
                        "label": str(label),
                        "split": normalized_split,
                        "image_path": str(image_path.resolve()),
                        "finding_labels": finding_labels,
                    }
                )

    if not rows:
        raise ValueError(
            "No images were found for the Kaggle pneumonia dataset layout. "
            "Expected folders like train/NORMAL, train/PNEUMONIA, val/NORMAL, val/PNEUMONIA, test/NORMAL, test/PNEUMONIA."
        )

    write_manifest(rows, output_manifest)
    return summarize_manifest(rows)


def verify_split_integrity(rows: list[dict[str, str]]) -> None:
    patient_splits: dict[str, set[str]] = {}
    for row in rows:
        patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
    leaking = [patient_id for patient_id, splits in patient_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"Patient-level leakage detected for {len(leaking)} patients.")


def verify_negative_labels(rows: list[dict[str, str]]) -> None:
    invalid = [
        row["image_id"]
        for row in rows
        if int(row["label"]) == 0 and row["finding_labels"].strip() != "No Finding"
    ]
    if invalid:
        raise ValueError(f"Found {len(invalid)} negative examples that are not clean No Finding cases.")


class ChestXRayDataset:
    def __init__(self, manifest_path: str | Path, split: str, image_size: int, train: bool, subset_size: int | None = None) -> None:
        Dataset, _ = _require_torchvision()
        Image, transforms = _require_image_stack()
        self._dataset_base = Dataset
        self._image_cls = Image
        self.rows = [row for row in load_manifest(manifest_path) if row["split"] == split]
        if subset_size is not None:
            self.rows = self.rows[:subset_size]
        self.transform = self._build_transform(transforms, image_size=image_size, train=train)

    def _build_transform(self, transforms: object, image_size: int, train: bool) -> object:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if train:
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[object, object, dict[str, str]]:
        import torch

        row = self.rows[index]
        image = self._image_cls.open(row["image_path"]).convert("RGB")
        tensor = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label, row


def create_dataloader(manifest_path: str | Path, split: str, image_size: int, batch_size: int, num_workers: int, train: bool, subset_size: int | None = None, shuffle: bool | None = None) -> object:
    _, DataLoader = _require_torchvision()
    dataset = ChestXRayDataset(
        manifest_path=manifest_path,
        split=split,
        image_size=image_size,
        train=train,
        subset_size=subset_size,
    )
    do_shuffle = train if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def manifest_label_ratio(rows: list[dict[str, str]]) -> float:
    labels = np.array([int(row["label"]) for row in rows], dtype=np.float32)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0:
        return 1.0
    return float(negatives / positives)
