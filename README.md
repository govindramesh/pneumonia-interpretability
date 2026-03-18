# CS7643 Pneumonia Midterm Pipeline

This repository contains a reproducible implementation scaffold for the CS7643 milestone project:
binary pneumonia detection on `ChestX-ray14` with baselines, evaluation, and interpretability.

## What is included

- `prepare_data.py`: build a filtered, patient-level split manifest
- `train.py`: train `cnn`, `resnet50`, or `dinov2_linear`
- `evaluate.py`: compute metrics and generate report-ready artifacts
- `interpret.py`: create Grad-CAM/DINO maps and faithfulness reports
- `configs/`: example JSON configs for CNN, ResNet-50, DINOv2, and a small smoke test

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

## Midterm report checklist

The commands below map directly to the `Experiments & Results` section requirements for the
midterm report. You can assume GPU is available when running the real experiments; the configs
use `device: "auto"`, which will pick CUDA if PyTorch can see it.

### 1. Established dataset or data-collection pipeline specified

Use this command to build the filtered, patient-level split manifest that defines the dataset
pipeline for the report:

```bash
python3 prepare_data.py \
  --metadata-csv /path/to/Data_Entry_2017_v2020.csv \
  --image-root /path/to/images \
  --output-manifest artifacts/manifests/chestxray14_binary.csv
```

This gives you the dataset pipeline story for the report:

- source dataset: `ChestX-ray14`
- filtering rule: `Pneumonia` vs exact `No Finding`
- ambiguous multi-disease negatives dropped
- patient-level `train/val/test` split

After this runs, the dataset manifest will be at:

- `artifacts/manifests/chestxray14_binary.csv`

### 2. Baseline evaluated with proposed metrics

Run the two baseline models first.

Baseline 1: small CNN

```bash
python3 train.py --config configs/chestxray14_cnn.json
python3 evaluate.py --config configs/chestxray14_cnn.json
python3 interpret.py --config configs/chestxray14_cnn.json
```

Baseline 2: fine-tuned ResNet-50

```bash
python3 train.py --config configs/chestxray14_resnet50.json
python3 evaluate.py --config configs/chestxray14_resnet50.json
python3 interpret.py --config configs/chestxray14_resnet50.json
```

These runs will generate the baseline metrics and visuals you need:

- metrics JSON: `artifacts/experiments/<experiment_name>/metrics/test_metrics.json`
- prediction CSVs: `artifacts/experiments/<experiment_name>/predictions/`
- ROC/PR/confusion plots: `artifacts/experiments/<experiment_name>/plots/`
- explanation maps and faithfulness curves: `artifacts/experiments/<experiment_name>/interpretability/`

The proposed metrics already implemented are:

- `ROC-AUC`
- `PR-AUC`
- `F1`
- `precision`
- `recall / sensitivity`
- `specificity`
- confusion matrix

### 3. Results for your method (numbers/visuals preferred)

Run the upgraded method:

```bash
python3 train.py --config configs/chestxray14_dinov2_linear.json
python3 evaluate.py --config configs/chestxray14_dinov2_linear.json
python3 interpret.py --config configs/chestxray14_dinov2_linear.json
```

To generate side-by-side interpretability comparisons for the report, compare DINOv2 against
the ResNet baseline on the same test images:

```bash
python3 interpret.py \
  --config configs/chestxray14_dinov2_linear.json \
  --comparison-config configs/chestxray14_resnet50.json
```

This gives you:

- final numbers for the proposed method
- DINO explanation maps
- faithfulness reports
- side-by-side DINO vs ResNet explanation figures

### Recommended run order for the report

```bash
python3 prepare_data.py --metadata-csv /path/to/Data_Entry_2017_v2020.csv --image-root /path/to/images --output-manifest artifacts/manifests/chestxray14_binary.csv
python3 train.py --config configs/chestxray14_cnn.json
python3 evaluate.py --config configs/chestxray14_cnn.json
python3 train.py --config configs/chestxray14_resnet50.json
python3 evaluate.py --config configs/chestxray14_resnet50.json
python3 train.py --config configs/chestxray14_dinov2_linear.json
python3 evaluate.py --config configs/chestxray14_dinov2_linear.json
python3 interpret.py --config configs/chestxray14_resnet50.json
python3 interpret.py --config configs/chestxray14_dinov2_linear.json --comparison-config configs/chestxray14_resnet50.json
```

## Visuals to include at the bottom of the midterm report

Once the runs are complete, these are the visuals that should be inserted into the report:

- Dataset pipeline figure:
  class balance by split from `artifacts/experiments/<experiment_name>/plots/class_balance.png`
- Baseline results figure:
  ROC curve, PR curve, and confusion matrix from the strongest baseline run
- Proposed method results figure:
  ROC curve, PR curve, and confusion matrix from the DINOv2 run
- Qualitative error analysis:
  examples referenced from `top_correct.csv` and `top_errors.csv`
- Interpretability figure:
  one Grad-CAM example from the ResNet run and one DINO map from the DINOv2 run
- Comparison figure:
  side-by-side ResNet vs DINO explanation outputs from `interpretability/comparisons/`
- Faithfulness figure:
  deletion/confidence-drop curves from the `interpretability/` directory
- Metrics table:
  CNN vs ResNet-50 vs DINOv2 using the values from each `test_metrics.json`

## Notes

- The defaults are CPU-safe and use `num_workers=0`, but `device: "auto"` will use GPU when available.
- `DINOv2` loading uses `torch.hub`, so it may require network access unless cached locally.
- The smoke config intentionally uses a very small subset so the entire pipeline is testable without a GPU.
