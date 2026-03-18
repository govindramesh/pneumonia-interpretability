from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    manifest_path: str
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    train_subset_size: int | None = None
    eval_subset_size: int | None = None


@dataclass
class ModelConfig:
    name: str
    pretrained: bool = True
    freeze_backbone: bool = False
    gradcam_target_layer: str | None = None
    local_weights_path: str | None = None


@dataclass
class TrainingConfig:
    device: str = "auto"
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    loss_name: str = "bce"
    early_stopping_patience: int = 3
    save_every_epoch: bool = False


@dataclass
class EvaluationConfig:
    threshold_metric: str = "f1"
    save_predictions: bool = True
    save_plots: bool = True


@dataclass
class InterpretationConfig:
    num_examples: int = 8
    mask_fraction: float = 0.2
    curve_steps: int = 10


@dataclass
class OutputConfig:
    root_dir: str = "artifacts/experiments"


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    interpretation: InterpretationConfig = field(default_factory=InterpretationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_name=payload["experiment_name"],
            seed=payload.get("seed", 42),
            dataset=DatasetConfig(**payload["dataset"]),
            model=ModelConfig(**payload["model"]),
            training=TrainingConfig(**payload.get("training", {})),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
            interpretation=InterpretationConfig(**payload.get("interpretation", {})),
            output=OutputConfig(**payload.get("output", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def experiment_dir(self) -> Path:
        return Path(self.output.root_dir) / self.experiment_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "dataset": self.dataset.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "interpretation": self.interpretation.__dict__,
            "output": self.output.__dict__,
        }

    def save_resolved(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
