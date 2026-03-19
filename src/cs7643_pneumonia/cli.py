from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .data import (
    load_manifest,
    prepare_chestxray14_manifest,
    prepare_hf_chestxray14_manifest,
    verify_negative_labels,
    verify_split_integrity,
)
from .runner import compare_interpretability, evaluate_experiment, interpret_experiment, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CS7643 pneumonia project pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare_data", help="Build a filtered patient-level manifest")
    prepare_parser.add_argument("--metadata-csv")
    prepare_parser.add_argument("--image-root")
    prepare_parser.add_argument("--hf-dataset")
    prepare_parser.add_argument("--hf-output-image-dir")
    prepare_parser.add_argument("--hf-splits", default="train,valid,test")
    prepare_parser.add_argument("--no-streaming", action="store_true")
    prepare_parser.add_argument("--limit-per-class", type=int, default=None)
    prepare_parser.add_argument("--output-manifest", required=True)
    prepare_parser.add_argument("--seed", type=int, default=42)

    for name, help_text in (
        ("train", "Train an experiment from JSON config"),
        ("evaluate", "Evaluate a saved checkpoint from JSON config"),
        ("interpret", "Generate interpretability outputs from JSON config"),
    ):
        command_parser = subparsers.add_parser(name, help=help_text)
        command_parser.add_argument("--config", required=True)
        command_parser.add_argument("--checkpoint", default=None)
        if name == "interpret":
            command_parser.add_argument("--comparison-config", default=None)
            command_parser.add_argument("--comparison-checkpoint", default=None)

    return parser


def prepare_data_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(["prepare_data"] + (argv or []))
    if args.hf_dataset:
        if not args.hf_output_image_dir:
            raise SystemExit("--hf-output-image-dir is required when using --hf-dataset.")
        summary = prepare_hf_chestxray14_manifest(
            hf_dataset=args.hf_dataset,
            output_manifest=args.output_manifest,
            output_image_dir=args.hf_output_image_dir,
            seed=args.seed,
            hf_splits=tuple(item.strip() for item in args.hf_splits.split(",") if item.strip()),
            streaming=not args.no_streaming,
            limit_per_class=args.limit_per_class,
        )
    else:
        if not args.metadata_csv or not args.image_root:
            raise SystemExit("--metadata-csv and --image-root are required unless --hf-dataset is used.")
        summary = prepare_chestxray14_manifest(
            metadata_csv=args.metadata_csv,
            image_root=args.image_root,
            output_manifest=args.output_manifest,
            seed=args.seed,
        )
    rows = load_manifest(args.output_manifest)
    verify_split_integrity(rows)
    verify_negative_labels(rows)
    print(f"Manifest written to {Path(args.output_manifest).resolve()}")
    for split, values in summary.items():
        print(f"{split}: total={values['total']} pos={values['positive']} neg={values['negative']} patients={values['patients']}")


def train_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(["train"] + (argv or []))
    config = ExperimentConfig.from_json(args.config)
    metrics = train_experiment(config)
    print(f"Training complete for {config.experiment_name}")
    print(metrics)


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(["evaluate"] + (argv or []))
    config = ExperimentConfig.from_json(args.config)
    metrics = evaluate_experiment(config, checkpoint_path=args.checkpoint)
    print(f"Evaluation complete for {config.experiment_name}")
    print(metrics)


def interpret_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(["interpret"] + (argv or []))
    config = ExperimentConfig.from_json(args.config)
    reports = interpret_experiment(config, checkpoint_path=args.checkpoint)
    if args.comparison_config:
        comparison_config = ExperimentConfig.from_json(args.comparison_config)
        generated = compare_interpretability(
            primary_config=config,
            comparison_config=comparison_config,
            primary_checkpoint_path=args.checkpoint,
            comparison_checkpoint_path=args.comparison_checkpoint,
        )
        print(f"Saved {len(generated)} side-by-side comparison figures.")
    print(f"Interpretation complete for {config.experiment_name}")
    print(f"Saved {len(reports)} explanation reports.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare_data":
        extra_args = [
            "--output-manifest",
            args.output_manifest,
            "--seed",
            str(args.seed),
        ]
        if getattr(args, "metadata_csv", None):
            extra_args += ["--metadata-csv", args.metadata_csv]
        if getattr(args, "image_root", None):
            extra_args += ["--image-root", args.image_root]
        if getattr(args, "hf_dataset", None):
            extra_args += ["--hf-dataset", args.hf_dataset]
        if getattr(args, "hf_output_image_dir", None):
            extra_args += ["--hf-output-image-dir", args.hf_output_image_dir]
        if getattr(args, "hf_splits", None):
            extra_args += ["--hf-splits", args.hf_splits]
        if getattr(args, "no_streaming", False):
            extra_args += ["--no-streaming"]
        if getattr(args, "limit_per_class", None) is not None:
            extra_args += ["--limit-per-class", str(args.limit_per_class)]
        prepare_data_main(extra_args)
    elif args.command == "train":
        train_main(["--config", args.config] + (["--checkpoint", args.checkpoint] if args.checkpoint else []))
    elif args.command == "evaluate":
        evaluate_main(["--config", args.config] + (["--checkpoint", args.checkpoint] if args.checkpoint else []))
    elif args.command == "interpret":
        extra_args = ["--config", args.config] + (["--checkpoint", args.checkpoint] if args.checkpoint else [])
        if getattr(args, "comparison_config", None):
            extra_args += ["--comparison-config", args.comparison_config]
        if getattr(args, "comparison_checkpoint", None):
            extra_args += ["--comparison-checkpoint", args.comparison_checkpoint]
        interpret_main(extra_args)
