# 3LC × HackBlox Scene Classification Challenge (Round 2)
### Data-Centric AI with ResNet-18 and 3LC

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework 3LC](https://img.shields.io/badge/3LC-2.22.3-green.svg)](https://3lc.ai)
[![Architecture](https://img.shields.io/badge/Model-ResNet18%20(From%20Scratch)-orange.svg)](https://pytorch.org)
[![Leaderboard](https://img.shields.io/badge/Public%20LB-Rank%20%231%20(0.77777)-gold.svg)]()
[![Reproducibility](https://img.shields.io/badge/Seed-Deterministic%20(42)-purple.svg)]()

---

## 1. Executive Summary & Challenge Overview

This project implements an end-to-end **Data-Centric AI** pipeline for the **HackBlox 2026 AI Track: 3LC Scene Classification Challenge**. The objective is to classify 1,800 unseen natural and urban scene images into six categories:
- `0: buildings`
- `1: forest`
- `2: glacier`
- `3: mountain`
- `4: sea`
- `5: street`

### Strict Challenge Constraints & Architecture Compliance
- **Model Architecture**: ResNet-18 **trained strictly from scratch** (`weights=None`). Zero external data, zero pretrained weights.
- **Dataset Budget**: 600 seed labeled images (100/class, balanced) from an unlabeled pool of 6,000 images (`undefined`).
- **Hard Labeling Constraint**: The final training table must strictly contain **at most 3,000 active samples (`weight = 1.0`)**. Our final table uses **2,800 active samples** (100% compliant).
- **Lineage Requirement**: At least three distinct active learning loops versioned inside 3LC (`train_0000`, `train_0001`, `train_0002`).
- **Hard-Negative Mining**: Loop 3 must be a dedicated hard-negative mining pass targeting the worst confused classes (`glacier` $\leftrightarrow$ `mountain` $\leftrightarrow$ `sea` and `buildings` $\leftrightarrow$ `street`).

---

## 2. Experimentation & Progression Summary

| Loop | Active Samples | Selection Strategy | 3LC Table URL / Revision | Val Accuracy | Public LB | Key Confusion / Diagnostic Milestone |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0 (Baseline)** | 600 / 3,000 | Baseline Seed Only | `intel-scene/tables/train` | **69.42%** | Submitted | Baseline floor. High confusion: glacier/mountain/sea (160+ cross errors) and buildings/street. |
| **1** | 1,400 / 3,000 | Margin Uncertainty + UMAP Diversity | `intel-scene/tables/train_0000` | **78.00%** | **0.77777 (Rank #1)** | **+8.58%** leap. Forest precision reached 94.4%. Glacier/mountain remained bottleneck. |
| **2** | 2,200 / 3,000 | Class-Balanced Error-Focused Sampling | `intel-scene/tables/train_0001` | **81.75%** | Pending | **+3.75%** leap. Mountain recall jumped to 75.5%, glacier to 69.5%, street F1 to 86.7%. |
| **3 (Final)** | **2,800 / 3,000** | Dedicated Hard-Negative Mining Pass | `intel-scene/tables/train_0002` | **82.00%** | Ready | Boundary pairs resolved. Sea recall reached 87.5%, forest recall 96.0%, glacier precision 85.1%. |

```
Validation Accuracy Trajectory Across Loops:
  Baseline (Loop 0): [=======================>                     ] 69.42% (600 samples)
  Loop 1:            [=============================>               ] 78.00% (1,400 samples)  <-- #1 on Kaggle LB (0.77777)
  Loop 2:            [=================================>           ] 81.75% (2,200 samples)
  Loop 3:            [==================================>          ] 82.00% (2,800 samples)
```

---

## 3. 3LC Table Lineage & Deliverables Architecture

```text
├── data/
│   ├── train/                 # 600 seed labeled (100/class) + 6000 undefined pool
│   ├── val/                   # 1200 balanced validation images (200/class)
│   └── test/                  # 1800 flat evaluation images
├── configs/
│   └── config.yaml            # Single source of truth for all hyperparameters
├── src/
│   ├── model.py               # ResNet-18 (from scratch, weights=None)
│   ├── dataset.py             # Flat test dataset and 3LC table mappers
│   ├── augment.py             # Domain-specific natural scene augmentations
│   ├── utils.py               # Seed fixing (42), confusion matrices, evaluation metrics
│   ├── register_tables.py     # Idempotent 3LC table registration
│   ├── train.py               # Deterministic training pipeline with 3LC integration
│   └── predict.py             # Inference generator aligned to Kaggle format
├── 3lc_tables/                # Exported table metadata and lineage records
├── logs/
│   ├── loop_log.md            # Detailed audit trail of each data-centric iteration
│   └── metrics.csv            # Run-by-run training and validation metrics
├── reports/                   # Saved confusion matrices, classification reports, embeddings
│   ├── final_writeup.md       # Comprehensive technical report
│   ├── loop0/                 # Baseline reports
│   ├── loop1/                 # Loop 1 reports
│   ├── loop2/                 # Loop 2 reports
│   └── loop3/                 # Loop 3 reports
├── submissions/               # Timestamped historical submission files
├── submission.csv             # Final verified submission (1800 rows)
├── 3lc_project_Intel-Scene.zip# Zipped 3LC project directory (~/.local/share/3LC/projects/Intel-Scene)
├── sample_submission.csv      # Ground-truth format template (1800 rows)
├── requirements.txt           # Pinned dependencies
└── README.md                  # Challenge documentation
```

### 3LC Table Lineage Tree:
```
intel-scene/tables/train (600 active)
    └── intel-scene/tables/train_0000 (1,400 active, Loop 1)
            └── intel-scene/tables/train_0001 (2,200 active, Loop 2)
                    └── intel-scene/tables/train_0002 (2,800 active, Loop 3)
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

### Reproduction Protocol:
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reproduce Baseline (Loop 0)
python train.py --loop 0

# 4. Reproduce Loop 1
python train.py --loop 1 --advanced

# 5. Reproduce Loop 2
python train.py --loop 2 --advanced

# 6. Reproduce Final Loop 3
python train.py --loop 3 --advanced

# 7. Generate final Kaggle submission
python predict.py
```

---

## 5. Official Submission Checklist for Judges

- [x] **Architecture**: ResNet-18 initialized strictly from scratch (`torchvision.models.resnet18(weights=None)`).
- [x] **Zero External Data**: Trained only on provided competition images and curated pool labels.
- [x] **Sample Budget**: Exactly 2,800 active rows ($\le 3,000$ limit).
- [x] **3LC Table Lineage**: Versioned tables `train`, `train_0000`, `train_0001`, `train_0002` present and linked in 3LC.
- [x] **Zipped 3LC Archive**: `3lc_project_Intel-Scene.zip` created and ready for inspection.
- [x] **1,800 Row Submission**: `submission.csv` validated with 100% ID alignment, zero nulls, and calibrated class probabilities.
- [x] **Full Technical Report**: Comprehensive documentation available at `reports/final_writeup.md`.
