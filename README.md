# CS7643 Pneumonia Midterm Pipeline

This repository contains a CPU-friendly implementation scaffold for the CS7643 milestone project:
binary pneumonia detection on `ChestX-ray14` with baselines, evaluation, and interpretability.

## What is included

- `prepare_data.py`: build a filtered, patient-level split manifest
- `train.py`: train `cnn`, `resnet50`, or `dinov2_linear`
- `evaluate.py`: compute metrics and generate report-ready artifacts
- `interpret.py`: create Grad-CAM/DINO maps and faithfulness reports
- `configs/`: example JSON configs, including a small smoke config

## Installation

Create a virtual environment and install the minimal dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected data inputs

`prepare_data.py` expects:

- the ChestX-ray14 metadata CSV, such as `Data_Entry_2017_v2020.csv`
- an image root containing the image files referenced by `Image Index`

The script filters:

- positive class: any example whose labels contain `Pneumonia`
- negative class: examples labeled exactly `No Finding`
- all other cases are dropped

## Example workflow

1. Build the manifest:

```bash
python3 prepare_data.py \
  --metadata-csv /path/to/Data_Entry_2017_v2020.csv \
  --image-root /path/to/images \
  --output-manifest artifacts/manifests/chestxray14_binary.csv
```

2. Run a smoke experiment on CPU:

```bash
python3 train.py --config configs/chestxray14_smoke.json
```

3. Evaluate the saved checkpoint:

```bash
python3 evaluate.py --config configs/chestxray14_smoke.json
```

4. Produce explanations:

```bash
python3 interpret.py --config configs/chestxray14_smoke.json
```

## Notes

- The defaults are CPU-safe and use `num_workers=0`.
- `DINOv2` loading uses `torch.hub`, so it may require network access unless cached locally.
- The smoke config intentionally uses a very small subset so the entire pipeline is testable without a GPU.
