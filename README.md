# 3LC × HackBlox Scene Classification Challenge (Round 2)
### Data-Centric Active Learning, Regularization & Ensembling with ResNet-18 and 3LC

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework 3LC](https://img.shields.io/badge/3LC-2.22.3-green.svg)](https://3lc.ai)
[![Architecture](https://img.shields.io/badge/Model-ResNet18%20(From%20Scratch)-orange.svg)](https://pytorch.org)
[![Leaderboard](https://img.shields.io/badge/Public%20LB-Rank%20%231%20(0.77777)-gold.svg)]()
[![Validation OOF](https://img.shields.io/badge/CV%20OOF-81.25%25%20%C2%B1%201.93%25-brightgreen.svg)]()
[![Reproducibility](https://img.shields.io/badge/Seed-Deterministic%20(42)-purple.svg)]()

---

## 1. Executive Summary & Challenge Overview

This repository contains the complete, production-grade **Data-Centric AI** pipeline developed for the **HackBlox 2026 AI Track: 3LC Scene Classification Challenge**. The task is to accurately classify 1,800 unseen test scene images into six classes:
- `0: buildings`
- `1: forest`
- `2: glacier`
- `3: mountain`
- `4: sea`
- `5: street`

### Strict Challenge Constraints & Architecture Compliance
- **Model Architecture**: ResNet-18 **trained strictly from scratch** (`weights=None`). Zero external pretrained models, zero ImageNet weights, zero foundation vision models (e.g. CLIP).
- **Dataset Budget**: 600 seed labeled images (100 per class) from an unlabeled pool of 6,000 images (`undefined`).
- **Hard Labeling Constraint**: Final training tables must strictly contain **at most 3,000 active samples (`weight = 1.0`)**. Our final table uses **exactly 3,000 active samples** (100% compliant).
- **Lineage Requirement**: Versioned, immutable active learning cycles tracked inside 3LC (`train` $	o$ `train_0000` $	o \dots 	o$ `train_0005` $	o$ `train_0009`).
- **Hard-Negative Mining**: Targeted passes to resolve geological (`glacier` $\leftrightarrow$ `mountain`) and architectural (`buildings` $\leftrightarrow$ `street`) ambiguities.

---

## 2. Experimentation & Data-Centric Progression

| Loop / Table | Active Samples | Selection Strategy | 3LC Table Revision | Val Accuracy | Notes & Key Diagnostics |
|:---:|:---:|:---:|:---:|:---:|:---|
| **0 (Baseline)** | 600 / 3,000 | Baseline Seed Only | `intel-scene/tables/train` | **69.42%** | High cross-confusion between glacier, mountain, and sea (160+ errors). |
| **1** | 1,400 / 3,000 | Margin Sampling + UMAP Diversity | `intel-scene/tables/train_0000` | **78.00%** | **Rank #1 on Kaggle LB (0.77777)**. Forest precision jumped to 94.4%. |
| **2** | 2,200 / 3,000 | Class-Balanced Error Mining | `intel-scene/tables/train_0001` | **81.75%** | Refined mountain/glacier decision boundary. |
| **Curated Safe Floor** | 2,498 / 3,000 | Curated boundary samples | `intel-scene/tables/train_0005` | **78.42% (TTA)** | 100% indisputable human-curated safe fallback floor (`submission_train0005_safe.csv`). |
| **Full Active Budget** | 3,000 / 3,000 | Full 3,000 budget, expert boundary review | `intel-scene/tables/train_0009` | **78.67%** | Mountain recall 77.5%; glacier suffered from class imbalance. |
| **Inverse-Freq Sampler** | 3,000 / 3,000 | Inverse class frequency sampling | `intel-scene/tables/train_0009` | **79.83% (TTA)** | Recovered +6.0% glacier recall without sacrificing mountain. |
| **TrivialAugmentWide** | 3,000 / 3,000 | Dynamic augmentation policy | `intel-scene/tables/train_0009` | **79.42%** | Highest single-model score without ensembling. |
| **SWA (6 epochs)** | 3,000 / 3,000 | Stochastic Weight Averaging (lr=3e-5) | `intel-scene/tables/train_0009` | **81.00% (TTA)** | Flatter minima: single model reached 80.17% standard / 81.00% with 2-scale TTA. |
| **6-Way Grand Stack** | 3,000 / 3,000 | Temperature scaling + OOF calibration | `intel-scene/tables/train_0009` | **81.25% ± 1.93%** | Final Kaggle candidate (`submission.csv`). 5-fold cross-validated out-of-fold. |

```
Validation Accuracy Trajectory Across Progression:
  Baseline (Loop 0):  [=======================>                     ] 69.42% (600 samples)
  Loop 1:             [=============================>               ] 78.00% (1,400 samples)  <-- #1 on Kaggle LB (0.77777)
  Loop 2:             [=================================>           ] 81.75% (2,200 samples)
  Curated Safe Floor: [=============================>               ] 78.42% (2,498 samples)
  SWA + 2-Scale TTA:  [==================================>          ] 81.00% (3,000 samples)
  6-Way Grand Stack:  [===================================>         ] 81.25% ± 1.93% OOF (3,000 samples)
```

---

## 3. Key Technical Breakthroughs & Ablations

### A. The Glacier $\leftrightarrow$ Mountain Tradeoff
- **The Issue**: Glacier and snow-capped mountain rock share identical high-contrast textures. ERM loss favored the majority class (`mountain`), driving glacier recall down to 49.0%.
- **The Fix**: `WeightedRandomSampler` with inverse-frequency weights restored glacier recall to 55.0% standalone and 63.0% in the ensemble, maintaining 78.0% mountain recall.

### B. Directional Texture Sensitivity & 2-Scale Multi-Crop TTA
- **Ablation Finding**: Standard horizontal flip TTA **reduced** validation accuracy (-0.34%), because natural terrain features (sunlight shadowing, cliff angles) are directional.
- **The Solution**: 2-Scale Multi-Crop TTA strictly without horizontal flips:
  $$\text{Prediction} = 0.5 \times f(x_{\text{std } 150 \times 150}) + 0.5 \times f(\text{CenterCrop}_{150}(\text{Resize}_{160}(x)))$$
  This yielded an immediate **+0.83% lift** across all evaluated models.

### C. Stochastic Weight Averaging (SWA)
- Training ResNet-18 from scratch on small datasets is prone to sharp loss landscapes. 6 epochs of SWA with low learning rate ($3 \times 10^{-5}$) and BatchNorm re-estimation achieved **80.17% standalone / 81.00% with TTA**.

### D. Out-of-Fold Temperature-Scaled Calibration
- Temperature scaling ($z_i / T_i$) fitted on 5-fold out-of-fold cross-validation prevented overconfident models from distorting ensemble probabilities:
  - Equal-weights OOF: **80.92%**
  - Calibrated weighted OOF: **81.25% ± 1.93%** (Fold range: 78.33% – 83.33%)

---

## 4. Final Per-Class Performance (6-Way Grand Stack)

Evaluated via strict 5-fold cross-validation on 1,200 held-out validation samples:

| Class | Precision | Recall | F1-Score | Diagnostic Analysis |
|:---|:---:|:---:|:---:|:---|
| **forest (1)** | 0.94 | **95.5%** | 0.95 | Clean separation from natural vegetation. |
| **street (5)** | 0.87 | **90.0%** | 0.88 | Road asphalt & perspective cleanly resolved. |
| **buildings (0)** | 0.79 | **80.5%** | 0.80 | High geometric structural precision. |
| **sea (4)** | 0.81 | **80.5%** | 0.81 | Clear water/coastline boundaries. |
| **mountain (3)** | 0.76 | **78.0%** | 0.77 | Protected rock-face textures. |
| **glacier (2)** | 0.70 | **63.0%** | 0.66 | Massive recovery from initial 49.0% floor. |
| **Overall OOF** | — | — | **81.25% ± 1.93%** | Solid out-of-fold generalization across all folds. |

---

## 5. Two-Submission Strategy for Kaggle

Kaggle allows each team to select **two final submissions**. To ensure both maximum competitive upside and total safety against any technical scrutiny, we submit:

1. **Submission 1 (100% Uncontested Fallback Floor)**:
   - **File**: `submission_train0005_safe.csv`
   - **Lineage**: Trained purely on `train_0005` (2,498 active rows, 100% verified baseline).
   - **Validation Accuracy**: **78.42%** (3-seed ensemble with TTA).

2. **Submission 2 (Top Clean Generalization)**:
   - **File**: `submission.csv` (also saved as `submission_grand_stack_8125.csv`)
   - **Lineage**: 6-Way Grand Stack trained on `train_0005` + `train_0009` (3,000 active rows) with SWA, balanced sampling, 2-scale multi-crop TTA, and temperature calibration.
   - **Validation Accuracy**: **81.25% ± 1.93% OOF**.
   - **Balanced Test Class Distribution**:
     - `buildings`: 279 (15.5%)
     - `forest`: 301 (16.7%)
     - `glacier`: 226 (12.6%)
     - `mountain`: 337 (18.7%)
     - `sea`: 338 (18.8%)
     - `street`: 319 (17.7%)

---

## 6. Project Architecture & 3LC Archive

```text
├── 3lc_project_Intel-Scene.zip.part_aa # Split archive part 1 (80 MB)
├── 3lc_project_Intel-Scene.zip.part_ab # Split archive part 2 (74 MB)
├── 3lc_tables/                         # Exported 3LC table metadata & revision history
├── configs/
│   └── config.yaml                     # Single source of truth for hyperparameters
├── src/
│   ├── model.py                        # ResNet-18 (strictly from scratch, weights=None)
│   ├── dataset.py                      # Flat test dataset and 3LC table loaders
│   ├── augment.py                      # Domain augmentations & 2-Scale Multi-Crop TTA
│   ├── utils.py                        # Seed fixing (42), metrics, confusion matrices
│   ├── train.py                        # Full reproducible training pipeline
│   ├── ensemble.py                     # Temperature-scaled ensembling engine
│   └── predict.py                      # Test inference generator
├── scratch/                            # Diagnostic evaluation and training scripts
├── generate_grand_stack_submission.py  # Script generating the 81.25% final submission
├── submission.csv                      # Final Primary Submission (1,800 rows)
├── submission_train0005_safe.csv       # Final Safe Fallback Submission (1,800 rows)
├── sample_submission.csv               # Ground-truth format template (1,800 rows)
├── reports/
│   └── final_writeup.md                # Full technical writeup & Google Form answers
└── README.md
```

### Reassembling the 3LC Project Archive
To comply with GitHub's 100 MB single-file limit without requiring Git LFS, the 154 MB 3LC project directory archive is stored in two parts. Reassemble it with:
```bash
cat 3lc_project_Intel-Scene.zip.part_* > 3lc_project_Intel-Scene.zip
sha256sum 3lc_project_Intel-Scene.zip
# Expected SHA256: 28cd4ba4493e5d634f85cb967810b62800a8b3bf22c4d1fb9a35975e8e0831ca
```

---

## 7. Reproduction Protocol

```bash
# 1. Setup Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Reassemble 3LC Project Archive
cat 3lc_project_Intel-Scene.zip.part_* > 3lc_project_Intel-Scene.zip

# 3. Generate Final Grand Stack Submission
python generate_grand_stack_submission.py
# Produces submission.csv (1,800 rows) matching sample_submission.csv format
```

---

## 8. Official Submission & Compliance Checklist

- [x] **Architecture**: ResNet-18 initialized strictly from scratch (`torchvision.models.resnet18(weights=None)`).
- [x] **Zero External Weights / Models**: No pretrained checkpoints, zero CLIP or foundation models.
- [x] **Sample Budget**: Exactly 3,000 active rows (<= 3,000 limit).
- [x] **3LC Table Lineage**: Complete auditable lineage inside 3LC (`train` -> ... -> `train_0009`).
- [x] **Zipped 3LC Archive**: `3lc_project_Intel-Scene.zip.part_aa` and `.part_ab` present in repository.
- [x] **Submission Alignment**: Exactly 1,800 rows, zero nulls, perfectly aligned image IDs.
- [x] **Evaluation Form**: Complete responses documented in `reports/final_writeup.md`.
- [x] **Collaborator Access**: Added judge `Rishikesh-Jadhav` on GitHub.
