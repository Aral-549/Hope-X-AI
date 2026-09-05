# 3LC × HackBlox Scene Classification Challenge (Round 2)
### Data-Centric AI with ResNet-18 and 3LC

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework 3LC](https://img.shields.io/badge/3LC-2.22.3-green.svg)](https://3lc.ai)
[![Architecture](https://img.shields.io/badge/Model-ResNet18%20(From%20Scratch)-orange.svg)](https://pytorch.org)
[![Reproducibility](https://img.shields.io/badge/Seed-Deterministic%20(42)-purple.svg)]()

---

## 1. Executive Summary & Challenge Overview

This project implements an end-to-end **Data-Centric AI** pipeline for the **HackBlox 2026 AI Track: 3LC Scene Classification Challenge**. The objective is to build a high-performance 6-class natural scene classifier:
- `0: buildings`
- `1: forest`
- `2: glacier`
- `3: mountain`
- `4: sea`
- `5: street`

### The Data-Centric Twist
Unlike standard deep learning competitions where participants iterate on complex model architectures or fine-tune massive pretrained foundation models, this challenge strictly fixes the architecture:
- **Model Architecture**: ResNet-18 **trained strictly from scratch** (`weights=None`). No pretrained weights allowed.
- **Dataset Budget**: 600 seed labeled images (100 per class, balanced) + 6,000 unlabeled pool (`undefined`).
- **Hard Labeling Constraint**: The final training table may contain **at most 3,000 active samples (`weight = 1`)**, including the 600 initial seed labels.
- **Evaluation Metric**: Classification accuracy on 1,800 unseen test images (50% public leaderboard / 50% private leaderboard).

Accuracy gains are driven entirely by **systematic data curation using 3LC**: embeddings-guided active learning, hard negative mining, sample reweighting, and cleaning label noise.

---

## 2. Clean Repository Architecture

```text
├── data/
│   ├── train/                 # 600 seed labeled (100/class) + 6000 undefined pool
│   ├── val/                   # 1200 balanced validation images (200/class)
│   └── test/                  # 1800 flat evaluation images
├── configs/
│   └── config.yaml            # Single source of truth for all hyperparameters
├── src/
│   ├── __init__.py
│   ├── model.py               # Fixed ResNet-18 (from scratch)
│   ├── dataset.py             # Flat test dataset and 3LC table mappers
│   ├── augment.py             # Domain-specific natural scene augmentations
│   ├── utils.py               # Seed fixing, confusion matrices, evaluation metrics
│   ├── register_tables.py     # Idempotent 3LC table registration
│   ├── train.py               # Deterministic training pipeline with 3LC integration
│   └── predict.py             # Inference generator aligned to Kaggle format
├── 3lc_tables/                # Exported table metadata and lineage records
├── logs/
│   ├── loop_log.md            # Detailed audit trail of each data-centric iteration
│   └── metrics.csv            # Run-by-run training and validation metrics
├── reports/                   # Saved confusion matrices, classification reports, embeddings
│   ├── loop0_baseline/
│   ├── loop1/
│   ├── loop2/
│   └── loop3/
├── notebooks/                 # Exploratory notebooks and error analysis
├── register_tables.py         # Root entrypoint
├── train.py                   # Root entrypoint
├── predict.py                 # Root entrypoint
├── sample_submission.csv      # Ground-truth format template (1800 rows)
├── requirements.txt           # Pinned dependencies
└── README.md                  # Comprehensive challenge documentation
```

---

## 3. Environment Setup & Pinned Dependencies

Ensure Python 3.11 is used. Install the exact pinned dependencies:

```bash
# 1. Create and activate virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# 2. Install PyTorch and dependencies
pip install -r requirements.txt
```

### 3LC Account & Authentication
```bash
# Login to 3LC platform
3lc login <your_api_key>

# Start local 3LC background service (required for 3LC Dashboard)
3lc service
```

---

## 4. Reproducibility & Seeding Guarantee

Determinism is enforced in `src/utils.py` via `set_seed(42)`:
- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `os.environ["PYTHONHASHSEED"] = "42"`

### Seeding Verification Check
Running the pipeline twice from a clean environment produces identical validation metrics and model state representations, confirming reproducibility.

---

## 5. End-to-End Execution Protocol

### Step 1: Register Initial 3LC Tables
```bash
python register_tables.py
```
Initializes versioned 3LC tables (`train` and `val`) referencing dataset images. Labeled seed images receive `weight = 1.0`, while undefined pool images receive `weight = 0.0`.

### Step 2: Train Phase 1 Baseline
```bash
python train.py --loop 0
```
Trains ResNet-18 on the 600 seed samples only. Generates `best_model.pth`, calculates baseline confusion matrix, and extracts 3D UMAP embeddings into the 3LC Dashboard.

### Step 3: Generate Early Submission
```bash
python predict.py
```
Generates verified `submission.csv` aligned with `sample_submission.csv` to lock in an immediate baseline score on the Kaggle leaderboard.

### Step 4: Iterative Labeling Loops (Loops 1–3)
1. Launch the 3LC Dashboard: `3lc service`
2. Inspect embedding clusters, class confusion boundaries (specifically glacier ↔ mountain and street ↔ buildings).
3. Apply active learning criteria (uncertainty sampling, boundary mining, hard-negative selection).
4. Label high-value samples and save a new versioned table revision.
5. Retrain:
   ```bash
   python train.py --loop 1 --advanced
   python predict.py
   ```
6. Record metrics in `logs/loop_log.md` and `logs/metrics.csv`.

---

## 6. Official Submission Checklist

- [x] ResNet-18 architecture strictly initialized from scratch (`weights=None`).
- [x] Zero external data or pretrained weights used.
- [x] Total active training samples verified $\le 3,000$.
- [x] All 1,800 test image predictions validated against `sample_submission.csv`.
- [x] Lineage preserved across versioned 3LC table revisions.
