# Hand Verification Training (HandNet)

Minimal, reproducible training pipeline for hand verification built with PyTorch. Works on Apple Silicon (MPS), CUDA, or CPU. Provides a small custom CNN or a ResNet18 backbone, cross-entropy or triplet loss, and EER/AUC evaluation with 2-image enrollment and cosine similarity.

## Features
- Simple CLI: one script to train/evaluate.
- Backbones: `resnet18` (ImageNet) or a small custom `handnet`.
- Losses: cross-entropy (ID classification) or triplet margin.
- Evaluation: EER/AUC with mean enrollment (2 images per subject).
- Artifacts: best weights, ROC plot, metrics, and raw scores.

## Repo Layout
```
.
├── scripts/
│   └── train_handnet.py        # Entry CLI for training/eval
├── ui/
│   ├── app.py                  # Streamlit main app
│   ├── models.py               # Models (Featured) tab
│   ├── library.py              # Library tab
│   ├── train_eval.py           # Train/Eval (Advanced) tab
│   ├── verify.py               # Verify tab
│   └── utils.py                # UI helpers
├── handnet/
│   ├── __init__.py             # Re-exports for convenient imports
│   ├── utils.py                # device, seeding, CSV filters, subject split, image index
│   ├── data.py                 # HandDataset and transforms
│   ├── models.py               # HandNetSmall, ResNet18 embedder, classifier head
│   ├── losses.py               # Triplet loss (batch-hard lite)
│   ├── train_loop.py           # train_one_epoch
│   └── eval.py                 # embeddings, EER/AUC, ROC, metrics save
├── configs/
│   ├── datasets/               # One YAML per dataset
│   │   └── palmar_clean.yaml
│   └── pipelines/              # One YAML per pipeline (recipe)
│       ├── res18_ce_to_triplet_pk16x4.yaml
│       └── handnet_ce_to_triplet_pk16x4.yaml
├── requirements.txt            # Pip deps
├── environment.yml             # Conda env (optional)
└── runs/                       # Output directory (created automatically)
```

## Requirements
- Python 3.9+ recommended
- PyTorch + TorchVision (installed via `requirements.txt` or `environment.yml`)
- Optional GPU: CUDA or Apple Silicon (MPS). The script auto-selects the best device.

## Setup
Using pip (recommended):
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using conda:
```
conda env create -f environment.yml
conda activate <env-name-from-yml>
```

## Data Preparation
- `--csv_path` should point to a CSV with at least these columns:
  - `imageName`: filename of the image (e.g., `IMG_0001.jpg`)
  - `id`: subject ID (integer or numeric string)
  - Optional quality columns (if present): `aspectOfHand`, `accessories`, `nailPolish`, `irregularities`
- `--data_dir` should point to the root folder containing images. Images can be nested in subfolders.
- The loader matches images by lowercased filename. Ensure filenames in the CSV uniquely identify images when lowercased.

## Quick Start
From the repo root:
```
# ResNet18 + Cross-Entropy on palmar
python scripts/train_handnet.py \
  --data_dir /path/to/images \
  --csv_path /path/to/HandInfo.csv \
  --backbone resnet18 --loss ce \
  --aspect palmar --epochs 15 --batch_size 64 --img_size 224 --device auto

# Small CNN + Triplet on dorsal
python scripts/train_handnet.py \
  --data_dir /path/to/images \
  --csv_path /path/to/HandInfo.csv \
  --backbone handnet --loss triplet \
  --aspect dorsal --epochs 25 --batch_size 96 --img_size 224 --device auto
```

Evaluate only (given weights):
```
python train_handnet.py \
  --data_dir /path/to/images \
  --csv_path /path/to/HandInfo.csv \
  --backbone resnet18 --eval_only --weights runs/<your_run>/best.pt \
  --aspect palmar --img_size 224 --device auto
```

## CLI Options
- `--data_dir`: Root directory containing images.
- `--csv_path`: Path to metadata CSV.
- `--aspect`: `palmar` | `dorsal` | `any` (filter by aspect if available).
- `--clean_only`: If set, drop rows flagged for `accessories`, `nailPolish`, and `irregularities` (if columns exist).
- `--min_images`: Keep subjects with at least this many images (default: 3).
- `--img_size`: Image size for training/eval transforms (default: 224).
- `--batch_size`: Batch size (default: 64).
- `--epochs`: Number of epochs (default: 15).
- `--seed`: Random seed (default: 42).
- `--backbone`: `resnet18` or `handnet`.
- `--loss`: `ce` or `triplet`.
- `--device`: `auto` | `cpu` | `cuda` | `mps`.
- `--eval_only`: Skip training and only evaluate given `--weights`.
- `--weights`: Path to a `.pt` file (for `--eval_only`).
- `--out_dir`: Output directory for run artifacts (default: `runs`).

## Outputs
Each run writes to `runs/<backbone>_<loss>_<aspect>_<timestamp>/`:
- `best.pt`: Best model state by dev EER (includes classifier state when CE).
- `scores.csv`: Test pair scores (label, score).
- `metrics.json`: Dev/Test EER and AUC.
- `roc.png`: ROC curve for the test set.

## Tips to Improve EER
- Use `--clean_only` if those quality flags exist in your CSV.
- Increase `--min_images` (e.g., 5) to ensure healthier enroll/probe splits.
- Try larger `--img_size` (256–320) if compute allows.
- Prefer `resnet18` with ImageNet weights for a strong baseline.
- Consider triplet loss when your batches contain multiple images per subject.

If you want, we can add: ImageNet input normalization, a cosine margin head (ArcFace/CosFace), and a P×K sampler for better triplet batches.

## Streamlit UI
- Launch: `streamlit run ui/app.py`
- Tabs:
  - Models: browse featured models by category.
  - Library: searchable table of all runs; view metrics, args, ROC, and scores.
  - Pipelines: manage datasets/pipelines YAMLs; Train tab consumes these.
  - Train / Eval: select dataset + pipeline and run; shows live logs.
  - Verify: simple similarity scoring with a required `.pt` model.

## Troubleshooting
- ModuleNotFoundError: Run commands from the repo root so Python can import `handnet/`.
- FileNotFoundError for images: Check that CSV `imageName` matches actual filenames (case-insensitive) under `--data_dir` and that names are unique when lowercased.
- CUDA/MPS not used: pass `--device cuda` or `--device mps` explicitly to force a device.
- No subjects after filtering: loosen `--aspect`, omit `--clean_only`, or reduce `--min_images`.

## Acknowledgements
Built with PyTorch, TorchVision, scikit-learn, and Matplotlib.
