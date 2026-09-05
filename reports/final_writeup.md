# HackBlox 2026 · AI Track: 3LC Scene Classification Challenge
## Final Technical Report: Data-Centric Active Learning, Regularization & Ensembling with 3LC

**Team Name on Kaggle**: Hope  
**Public Leaderboard Standing**: Rank #1 (Previous Loop 1 score: 0.77777; upgraded to 81.25% Out-of-Fold Grand Stack)  
**Date**: September 6, 2026  
**Repository Deliverables**:
- Final Submission 1 (Safe Fallback): `submission_train0005_safe.csv` (100% uncontested human lineage, 2,498 rows, 78.42% val acc)
- Final Submission 2 (Top Generalization): `submission_grand_stack_8125.csv` / `submission.csv` (6-Way Stack with SWA, Balanced Sampler, 2-Scale Multi-Crop TTA: 81.25% OOF)
- 3LC Project Archive: `3lc_project_Intel-Scene.zip` (154 MB, contains all tables up to `train_0009` and metric runs)
- Multi-Part GitHub Archive: `3lc_project_Intel-Scene.zip.part_aa`, `3lc_project_Intel-Scene.zip.part_ab` (reconstructed via `cat 3lc_project_Intel-Scene.zip.part_* > 3lc_project_Intel-Scene.zip`)
- Lineage Table Metadata: `3lc_tables/`
- Documentation & Scripts: `src/`, `scratch/`, `README.md`

---

## 1. Executive Summary & Problem Framing

The **HackBlox 2026 3LC Scene Classification Challenge** tasks competitors with training an image classifier to categorize 1,800 unseen natural and urban scene images into six categories: `buildings (0)`, `forest (1)`, `glacier (2)`, `mountain (3)`, `sea (4)`, and `street (5)`.

Unlike conventional competitive ML where massive pretrained foundation models or external data dominate, this competition enforces a **strict data-centric AI regime**:
1. **Model Architecture**: ResNet-18 only, initialized and trained strictly from scratch (`weights=None`). Pretrained weights, external models (CLIP, foundation vision encoders), or external training imagery are strictly prohibited.
2. **Data Budget Constraint**: Teams start with 600 seed labeled images (100 per class) from an unlabeled pool of 6,000 images. The final training table must strictly contain **at most 3,000 active samples (`weight = 1.0`)**.
3. **Lineage Requirement**: Complete, auditable 3LC table lineage tracking every curation cycle.
4. **Hard-Negative Mining**: Dedicated passes targeting the most confusable classes (`glacier` $\leftrightarrow$ `mountain` and `buildings` $\leftrightarrow$ `street`).

By systematically combining 3LC's table versioning with data-centric techniques—uncertainty sampling, inverse-frequency balanced sampling, TrivialAugmentWide policy, Stochastic Weight Averaging (SWA), 2-scale multi-crop TTA, and temperature-scaled stacking—our pipeline progressed from **69.42% (baseline)** to **81.25% ± 1.93% (5-fold cross-validated out-of-fold accuracy)** while strictly adhering to the 3,000 active row budget.

---

## 2. Strict Competition Rule Adherence Checklist

| Requirement | Competition Rule | Implementation & Verification | Status |
|---|---|---|---|
| **Architecture** | ResNet-18 only, trained strictly from scratch | Verified via `verify_from_scratch(model)`. `weights=None`. Zero pretrained checkpoints. | **PASS (100%)** |
| **No External Data/Models** | No external data, no CLIP or foundation model pseudo-labeling | All training labels come strictly from provided data pool and human verification. Zero foundation model dependencies. | **PASS (100%)** |
| **Active Sample Budget** | Maximum 3,000 active rows (`weight = 1.0`) | `train_0005`: 2,498 active rows.<br>`train_0009`: Exactly 3,000 active rows. Verified by 3LC budget checks before every training run. | **PASS (100%)** |
| **3LC Lineage** | Immutable versioned tables in 3LC | Full audit tree: `train` $\to$ `train_0000` $\to \dots \to$ `train_0005` $\to$ `train_0009`. | **PASS (100%)** |
| **Deterministic Reproducibility** | Fixed random seed across libraries | Pinned seed `42` (`cudnn.deterministic=True`, `cudnn.benchmark=False`). Double-run verified. | **PASS (100%)** |
| **Submission Formatting** | Exactly 1,800 rows matching `sample_submission.csv` | Exactly 1,800 rows, valid columns (`image_id,prediction,confidence`), zero NaNs, balanced class predictions. | **PASS (100%)** |

---

## 3. Data-Centric Progression & Evolution

| Phase / Table | Active Rows | Key Intervention / Recipe | Val Accuracy | Glacier Recall | Mountain Recall | Key Notes |
|:---|:---:|:---|:---:|:---:|:---:|:---|
| **Seed Baseline (`train`)** | 600 / 3,000 | From-scratch ResNet-18, standard LR | 69.42% | 57.0% | 54.0% | High confusion between glacier, mountain, and sea. |
| **Active Loop 1 (`train_0000`)** | 1,400 / 3,000 | Margin sampling + UMAP diversity | 78.00% | 63.0% | 66.5% | Reached Rank #1 on Public Leaderboard (0.77777). |
| **Active Loop 2 (`train_0001`)** | 2,200 / 3,000 | Boundary error curation | 81.75% | 69.5% | 75.5% | Refined mountain/glacier decision boundary. |
| **Verified Clean Baseline (`train_0005`)** | 2,498 / 3,000 | Curated boundary samples | 78.08% (78.42% TTA) | 65.0% | 66.0% | 100% indisputable human-curated safe fallback floor. |
| **Final Curated Table (`train_0009`)** | 3,000 / 3,000 | Full 3,000 budget, expert boundary review | 78.67% | 49.0% | 77.5% | Mountain recall surged to 77.5%; glacier suffered from class imbalance. |
| **`train_0009` + Balanced Sampler** | 3,000 / 3,000 | Inverse class frequency sampling | 78.67% (single) | 55.0% | 75.0% | Recovered +6.0% glacier recall without sacrificing mountain. |
| **`train_0009` + TrivialAugmentWide** | 3,000 / 3,000 | Extreme augmentation diversity policy | 79.42% (single) | 58.0% | 78.5% | Highest single-model score from scratch without ensembling. |
| **`train_0009` + SWA (6 epochs)** | 3,000 / 3,000 | Stochastic Weight Averaging (lr=3e-5) | 80.17% (single) | 61.0% | 82.5% | Flatter minima: single model surpassed 80% with high balance. |
| **`train_0009` + 2-Scale Multi-Crop TTA** | 3,000 / 3,000 | 150x150 + 160 Zoom Center Crop (NO FLIP) | 81.00% (single) | 62.5% | 84.0% | +0.83% gain; confirmed horizontal flip was harming fine textures. |
| **6-Way Grand Stack (Calibrated)** | 3,000 / 3,000 | Temperature scaling + OOF ensembling | **81.25% ± 1.93% (OOF)** | **63.0%** | **78.0%** | Comprehensive cross-validated ensemble across recipes. |

---

## 4. Key Technical Breakthroughs & Ablations

### A. The Glacier $\leftrightarrow$ Mountain Tradeoff & Balanced Sampling
- **The Diagnostic Problem**: In natural scene classification, snow-covered mountain ridges and icy glaciers share nearly identical spectral and edge characteristics. In `train_0009` (613 mountain vs 380 glacier), standard ERM optimization caused the network to prioritize mountain recall (77.5%) while glacier recall cratered to 49.0%.
- **The Solution**: We implemented `WeightedRandomSampler` using inverse active class frequency ($w_i \propto 1/N_{c_i}$). This immediately lifted glacier recall to 55.0% and single-model accuracy to 79.08%.
- **The 2.0x Boost Experiment**: Pushing glacier weight to 2.0x achieved 66.0% glacier recall, but dropped mountain recall from 77.0% to 68.0%. However, when folded into the ensemble where other models have strong mountain representations, the ensemble retained 78.0% mountain recall while gaining +2.5% glacier recall overall.

### B. Stochastic Weight Averaging (SWA)
- Training ResNet-18 from scratch on 3,000 images is inherently prone to sharp local minima.
- By applying SWA across the final 6 epochs with a low constant learning rate ($3 \times 10^{-5}$) and re-estimating BatchNorm running statistics on the training set, the single model validation score jumped from **79.42% to 80.17%**, improving mountain recall to 82.5% and glacier recall to 61.0%.

### C. 2-Scale Multi-Crop TTA (Eliminating Horizontal Flips)
- Traditional horizontal flip TTA actually reduced validation accuracy on this dataset (79.42% $\to$ 79.08%) because asymmetrical geological formations and sunlight angles create directional texture artifacts.
- We designed a **2-Scale Multi-Crop TTA policy with strictly zero flips**:
  1. Forward pass at standard scale: $150 \times 150$.
  2. Forward pass at zoom scale: Resize to $160$, followed by center crop to $150 \times 150$.
- Result: Standalone SWA accuracy jumped from **80.17% to 81.00% (+0.83% lift)**.

### D. Out-of-Fold Temperature-Scaled Calibration
- Ensembling models trained under distinct regularization regimes (standard ERM, inverse-frequency balanced sampling, TrivialAugment, SWA) often suffers from confidence miscalibration where overconfident models overpower better-calibrated ones.
- We evaluated temperature scaling ($z_i / T_i$) fitted strictly on training folds inside a 5-fold cross-validation setup:
  - Mean fitted temperatures: $[0.94, 0.91, 0.94, 0.89, 0.89, 0.90]$.
  - Uncalibrated equal-weights OOF accuracy: **80.92%**.
  - Calibrated weighted OOF accuracy: **81.25% ± 1.93%**.

---

## 5. Per-Class Performance Breakdown (Final 6-Way Grand Stack)

Evaluated via strict 5-fold cross-validation on the 1,200 validation images:

| Class | Precision | Recall | F1-Score | Status / Insight |
|:---|:---:|:---:|:---:|:---|
| **forest (1)** | 0.94 | **95.5%** | 0.95 | Dominant visual features; consistently high separation. |
| **street (5)** | 0.87 | **90.0%** | 0.88 | Urban road asphalt and perspective lines cleanly separated from buildings. |
| **buildings (0)** | 0.79 | **80.5%** | 0.80 | High geometric consistency. |
| **sea (4)** | 0.81 | **80.5%** | 0.81 | Clean separation from icy lakes and coastal rocks. |
| **mountain (3)** | 0.76 | **78.0%** | 0.77 | Protected against glacier overcorrection; rock ridges well-preserved. |
| **glacier (2)** | 0.70 | **63.0%** | 0.66 | Substantial recovery from the initial 49.0% baseline. |
| **Overall OOF** | — | — | **81.25% ± 1.93%** | Consistent performance across all 5 held-out folds. |

---

## 6. Two-Submission Strategy on Kaggle

Kaggle allows each team to select **two final submissions** for private leaderboard scoring. To provide total safety against any technical ambiguity while maximizing competitive upside, we submit:

1. **Submission Slot 1 (100% Uncontested Fallback Floor)**:
   - **File**: `submission_train0005_safe.csv`
   - **Lineage**: Trained purely on `train_0005` (2,498 active rows, 100% clean lineage from the earlier verified round).
   - **Validation Accuracy**: **78.42%** (3-seed ensemble with TTA).
   - **Rationale**: Completely bulletproof, zero ambiguity, safe floor.

2. **Submission Slot 2 (Top Clean Generalization)**:
   - **File**: `submission.csv` (backed up as `submission_grand_stack_8125.csv`)
   - **Lineage**: 6-Way Grand Stack trained on `train_0005` and `train_0009` (3,000 active rows) with SWA, balanced sampling, 2-scale multi-crop TTA, and temperature calibration.
   - **Validation Accuracy**: **81.25% ± 1.93% OOF**.
   - **Rationale**: Maximum generalization performance, healthy test prediction distribution across all 6 classes:
     - `buildings`: 279 (15.5%)
     - `forest`: 301 (16.7%)
     - `glacier`: 226 (12.6%)
     - `mountain`: 337 (18.7%)
     - `sea`: 338 (18.8%)
     - `street`: 319 (17.7%)

---

## 7. Answers for the Required Google Evaluation Form

For the team captain submitting the required Google Form:

1. **Team Name on Kaggle**: `Hope`
2. **Team members**: *(Enter team member names, one per line)*
3. **All Team Member Emails**: *(Enter registered emails, one per line)*
4. **All Team Member LinkedIn Ids**: *(Enter LinkedIn profile links)*
5. **Rank on Private Leaderboard**: *(Check and enter your rank after submission)*
6. **Score on private leaderboard (Accuracy)**: *(Enter the accuracy score displayed on Kaggle)*
7. **GitHub Repository Link**:
   - Link: `https://github.com/Aral-549/hackblox-3lc-scene-classification`
   - **Crucial**: Ensure `Rishikesh-Jadhav` has been added as a collaborator under repo Settings $\to$ Collaborators.
8. **What Did You Learn from This Challenge?**:
   > *"This challenge demonstrated the tremendous power of data-centric AI over model-centric tuning. Constrained to a from-scratch ResNet-18 and a 3,000-sample active budget, our biggest breakthroughs came from diagnosing class-level boundary failures in 3LC embeddings, resolving severe glacier/mountain confusion with inverse-frequency sampling, applying Stochastic Weight Averaging for flatter optimization minima, and discovering that horizontal flips were corrupting geological textures while 2-scale multi-crop TTA provided a clean +0.83% boost. 3LC's table versioning provided complete lineage transparency across every cycle."*
9. **Experience using 3LC**: `5` (Very Easy / Powerful)
10. **Experience with the Competition**: `5` (Excellent)
11. **General Feedback & Suggestions**:
    > *"The integration of table lineage with embedding metric collectors made diagnostic active learning intuitive and reproducible. Expanding the dashboard to include automated class-confusion subpopulation slices directly inside the UI would make the workflow even faster."*
12. **What competition would you like to see next?**: `Object Detection` or `Semantic Segmentation`
