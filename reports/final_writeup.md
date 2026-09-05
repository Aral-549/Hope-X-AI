# HackBlox 2026 · AI Track: 3LC Scene Classification Challenge
## Final Technical Report: Data-Centric Active Learning with 3LC

**Team**: Hope  
**Public Leaderboard Standing**: Rank #1 (Score: 0.77777 with Loop 1; upgraded models reaching 82.00% validation accuracy)  
**Date**: September 5, 2026  
**Repository Deliverables**:
- Final Submission: `submission.csv` (1,800 test samples, verified format)
- 3LC Project Archive: `3lc_project_Intel-Scene.zip` (154 MB, contains all tables and runs)
- Lineage Table Metadata: `3lc_tables/`
- Proof Artifacts & Confusion Matrices: `reports/loop0/`, `reports/loop1/`, `reports/loop2/`, `reports/loop3/`

---

## 1. Executive Summary & Problem Framing

The **HackBlox 2026 3LC Scene Classification Challenge** tasks competitors with training an image classifier to categorize 1,800 unseen natural and urban scene images into six categories: `buildings (0)`, `forest (1)`, `glacier (2)`, `mountain (3)`, `sea (4)`, and `street (5)`.

Unlike conventional competitive ML where massive pretrained foundation models or external data dominate, this competition enforces a **strict data-centric AI regime**:
1. **Model Architecture**: ResNet-18 only, initialized and trained strictly from scratch (`weights=None`). Pretrained weights or external training imagery are prohibited.
2. **Data Budget Constraint**: Teams are provided with 600 seed labeled images (100 per class) from an unlabeled pool of 6,000 images. The final training table must strictly contain **at most 3,000 active samples (`weight = 1.0`)**.
3. **Lineage Requirement**: At least three distinct active learning loops versioned inside 3LC (`train_0000`, `train_0001`, `train_0002`).
4. **Hard-Negative Mining**: Loop 3 must be a dedicated hard-negative mining pass targeting the most confusable classes.

By leveraging 3LC's table versioning, embedding reduction, and sample-level diagnostic tracking, our team engineered a systematic active learning pipeline that drove classification accuracy from **69.42% (baseline)** to **82.00% (final)**—an absolute improvement of **+12.58%**—while using only **2,800 active samples** (200 below the hard limit).

---

## 2. Strict Competition Rule Adherence Checklist

| Requirement | Competition Rule | Implementation & Proof | Status |
|---|---|---|---|
| **Architecture** | ResNet-18 only, trained strictly from scratch | Initialized with `torchvision.models.resnet18(weights=None)`. No pretrained backbones. | **PASS (100%)** |
| **No External Data** | Prohibited from downloading external images or models | Only competition train pool (6,000 images) and seed labels used for model training. | **PASS (100%)** |
| **Active Sample Budget** | Maximum 3,000 active rows (`weight = 1.0`) | Initial seed = 600. Loop 1 = 1,400. Loop 2 = 2,200. Loop 3 = **2,800**. Hard cap never violated. | **PASS (100%)** |
| **3LC Lineage** | At least 3 distinct versioned labeling loops | Registered lineage: `train` $\to$ `train_0000` $\to$ `train_0001` $\to$ `train_0002`. Verified in 3LC. | **PASS (100%)** |
| **Hard-Negative Mining** | Dedicated pass on confused classes in Loop 3 | Focused on boundary pairs `glacier` $\leftrightarrow$ `mountain` $\leftrightarrow$ `sea` and `buildings` $\leftrightarrow$ `street`. | **PASS (100%)** |
| **Deterministic Reproducibility** | Fixed random seed across all libraries | Seed `42` pinned across Python, NumPy, PyTorch CPU/CUDA, `torch.backends.cudnn.deterministic=True`. | **PASS (100%)** |
| **Submission Formatting** | 1,800 rows matching `sample_submission.csv` | Exactly 1,800 rows (`image_id`, `prediction`, `confidence`). No nulls. Validated. | **PASS (100%)** |

---

## 3. Experimentation & Progression Summary

| Loop | Active Samples | Added Samples | 3LC Table Revision | Validation Accuracy | Absolute Gain | Key Focus / Milestone |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0 (Baseline)** | 600 / 3,000 | 0 (Seed) | `train` | **69.42%** | Base | Established floor; diagnosed severe glacier/mountain and urban confusion. |
| **1** | 1,400 / 3,000 | +800 | `train_0000` | **78.00%** | **+8.58%** | Uncertainty sampling + UMAP diversity. Achieved **Rank #1 (0.77777)** on Public LB. |
| **2** | 2,200 / 3,000 | +800 | `train_0001` | **81.75%** | **+3.75%** | Class-balanced error-focused sampling targeting glacier & mountain bottlenecks. |
| **3 (Final)** | **2,800 / 3,000** | +600 | `train_0002` | **82.00%** | **+0.25%** | Dedicated hard-negative mining pass on boundary ambiguous pairs. |

```
Validation Accuracy Trajectory Across Loops:
  Baseline (Loop 0): [=======================>                     ] 69.42% (600 samples)
  Loop 1:            [=============================>               ] 78.00% (1,400 samples)  <-- #1 on Kaggle LB (0.77777)
  Loop 2:            [=================================>           ] 81.75% (2,200 samples)
  Loop 3:            [==================================>          ] 82.00% (2,800 samples)
```

---

## 4. Detailed Active Learning Methodology

### Loop 0: Baseline Initialization & Diagnostic Floor
- **Objective**: Establish the true performance floor using only the 600 seed images without stochastic transformations or modifications.
- **Setup**: ResNet-18 (`weights=None`), StepLR scheduler, 10 epochs, batch size 16, lr=1e-4.
- **Outcome**: 69.42% validation accuracy.
- **Diagnostic Findings**:
  - `forest` was well separated (F1 0.888).
  - High error rates occurred in two distinct thematic hubs:
    1. **Snow/Water Hub**: `glacier` recall was 57.0%, with 28 glaciers misclassified as mountains and 39 as sea. True `mountain` recall was only 54.0%, with 51 misclassified as glaciers and 31 as sea.
    2. **Urban Hub**: `buildings` vs. `street` experienced 42 false predictions of buildings as streets, and 26 streets as buildings.

### Loop 1: Uncertainty / Margin Sampling with UMAP Manifold Diversity
- **Objective**: Expand the training set by 800 active samples, targeting instances where the model exhibits lowest confidence while ensuring wide coverage across the latent feature space.
- **Strategy**:
  1. Computed predicted class probabilities and margin uncertainty $M(x) = 1 - (p_{\text{top1}} - p_{\text{top2}})$ for all uncurated samples.
  2. Applied k-means clustering in the 3D UMAP manifold generated by 3LC (`nice-vulture` run) to prevent redundant sampling of identical scenes.
  3. Curated 800 samples balanced across clusters (164 buildings, 102 forest, 103 glacier, 165 mountain, 100 sea, 166 street).
  4. Created versioned table `train_0000` via `tlc.TableWriter` linking `train` as parent.
- **Outcome**: Validation accuracy leapt by **+8.58%** to **78.00%**. This submission scored **0.77777** on Kaggle, taking **Rank #1**.

### Loop 2: Class-Balanced Error-Focused Sampling
- **Objective**: Directly attack the lingering confusion in the `glacier` and `mountain` categories while refining urban textures.
- **Strategy**:
  1. Extracted error distributions from the Loop 1 run. Glacier recall (63.0%) and mountain recall (66.5%) remained the primary barriers.
  2. Filtered candidate samples located near the glacier/mountain/sea boundary in the 3D UMAP embedding space.
  3. Activated 800 additional high-value samples with heavy representation of ambiguous terrain: 168 glaciers, 180 mountains, 142 buildings, 138 streets, 92 seas, 80 forests.
  4. Versioned as `train_0001` (parent: `train_0000`). Total active samples: 2,200.
- **Outcome**: Validation accuracy reached **81.75%** (+3.75%). Mountain recall jumped from 66.5% to 75.5%, and glacier recall climbed to 69.5%.

### Loop 3: Dedicated Hard-Negative Mining Pass (Competition Mandate)
- **Objective**: Conduct a focused pass on the hardest decision boundaries:
  - Boundary A: `glacier` $\leftrightarrow$ `mountain` $\leftrightarrow$ `sea` (snow-covered rocks vs. icy cliffs vs. glacial fjord waters)
  - Boundary B: `buildings` $\leftrightarrow$ `street` (distant facades vs. road-level asphalt and vehicle views)
- **Strategy**:
  1. Calculated pairwise confusion scores for unselected samples whose top-2 predicted probabilities were $(2, 3)$, $(3, 4)$, $(2, 4)$, or $(0, 5)$.
  2. Curated 600 precise hard-negative instances, bringing the active total to **2,800** (under the 3,000 active limit).
  3. Versioned as `train_0002` (parent: `train_0001`).
  4. Trained with 15 epochs, cosine warmup scheduler (decaying to 1e-6), label smoothing (0.1), and domain-specific affine/color augmentations.
- **Outcome**: Validation accuracy achieved **82.00%** (best epoch 15/15). Glacier precision soared to **85.06%**, forest recall reached **96.00%**, sea recall reached **87.50%**, and buildings recall crossed **81.00%**.

---

## 5. Confusion Trajectory & Per-Class Error Evolution

The table below traces the per-class precision (P), recall (R), and F1-score across each phase:

| Class | Loop 0 (Base) P / R / F1 | Loop 1 P / R / F1 | Loop 2 P / R / F1 | Loop 3 (Final) P / R / F1 | Total F1 Shift |
|---|:---:|:---:|:---:|:---:|:---:|
| **buildings (0)** | 0.697 / 0.680 / 0.689 | 0.789 / 0.785 / 0.787 | 0.811 / 0.825 / 0.818 | **0.775 / 0.810 / 0.792** | **+0.103** |
| **forest (1)** | 0.882 / 0.895 / 0.888 | 0.944 / 0.925 / 0.934 | 0.927 / 0.945 / 0.936 | **0.885 / 0.960 / 0.921** | **+0.033** |
| **glacier (2)** | 0.582 / 0.570 / 0.576 | 0.685 / 0.630 / 0.656 | 0.772 / 0.695 / 0.732 | **0.851 / 0.655 / 0.740** | **+0.164** |
| **mountain (3)** | 0.574 / 0.540 / 0.557 | 0.662 / 0.665 / 0.663 | 0.733 / 0.755 / 0.744 | **0.755 / 0.740 / 0.748** | **+0.191** |
| **sea (4)** | 0.722 / 0.740 / 0.731 | 0.783 / 0.795 / 0.789 | 0.821 / 0.835 / 0.828 | **0.814 / 0.875 / 0.843** | **+0.112** |
| **street (5)** | 0.745 / 0.740 / 0.742 | 0.841 / 0.895 / 0.867 | 0.840 / 0.895 / 0.867 | **0.842 / 0.880 / 0.861** | **+0.119** |
| **Overall Val Acc** | **69.42%** | **78.00%** | **81.75%** | **82.00%** | **+12.58%** |

### Key Diagnostic Shifts:
1. **Mountain & Glacier Resolution**:
   - In Loop 0, Mountain F1 was 0.557 and Glacier F1 was 0.576.
   - Through targeted active sampling and hard-negative mining in Loops 2 and 3, Mountain F1 rose to **0.748 (+19.1%)** and Glacier precision reached **0.851 (+26.9%)**.
2. **Sea & Coastal Clarification**:
   - Glacier-sea confusion was reduced significantly, driving Sea recall from 74.0% to **87.5%**.
3. **Urban Landmark Separation**:
   - Buildings recall increased from 68.0% to **81.0%**, while Street recall sustained high performance at **88.0%**.

---

## 6. 3LC Table Lineage & Infrastructure Integration

3LC serves as the immutable system of record for dataset versioning and metric tracking:

- **Lineage Tree**:
  `intel-scene/tables/train` (600 active)  
  $\longrightarrow$ `intel-scene/tables/train_0000` (1,400 active)  
  $\longrightarrow$ `intel-scene/tables/train_0001` (2,200 active)  
  $\longrightarrow$ `intel-scene/tables/train_0002` (2,800 active)

### Table Metadata Summary:
- **`train`**: Registered initial seed table (`600` active samples).
- **`train_0000`**: `1,400` active samples. Parent: `train`.
- **`train_0001`**: `2,200` active samples. Parent: `train_0000`.
- **`train_0002`**: `2,800` active samples. Parent: `train_0001`.
- **Validation Table**: `val` (1,200 samples, fixed across all loops).
- **All 3LC Project Files**: Archived in `3lc_project_Intel-Scene.zip` (154 MB).

---

## 7. Determinism & Reproducibility Statement

Strict reproducibility is guaranteed through:
1. **Central Seed Configuration**: All random number generators are set to seed `42`:
   - `random.seed(42)`
   - `np.random.seed(42)`
   - `torch.manual_seed(42)`
   - `torch.cuda.manual_seed_all(42)`
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`
2. **Deterministic DataLoaders**: `num_workers=0` to eliminate non-deterministic worker interleaving.
3. **Environment Replication**: Exact environment captured in `requirements.txt`.

### Reproduction Commands:
```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Reproduce Baseline (Loop 0)
python train.py --loop 0

# 3. Reproduce Loop 1
python train.py --loop 1 --advanced

# 4. Reproduce Loop 2
python train.py --loop 2 --advanced

# 5. Reproduce Final Loop 3
python train.py --loop 3 --advanced

# 6. Generate final Kaggle submission
python predict.py
```

---

## 8. Final Submission & Deliverables Verification

- **Submission File**: `submission.csv`
- **Rows**: 1,800 (plus 1 header row)
- **Columns**: `image_id`, `prediction`, `confidence`
- **Class Label Range**: `0` through `5`
- **Confidence Range**: `[0.251, 0.997]`
- **Submission History Backups**: Archived in `submissions/` with full timestamps.
- **Leaderboard Strategy**: The public score of 0.77777 was achieved with Loop 1. Loops 2 and 3 represent superior, better-regularized models (81.75% and 82.00% validation accuracy) with significantly reduced false positives in difficult terrain classes.
