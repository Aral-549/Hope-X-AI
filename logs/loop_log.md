# HackBlox 2026 · 3LC Data-Centric AI Challenge
## Iterative Labeling & Training Experiment Log

| Loop | Timestamp | Selection Criterion | Samples Touched | Cum. Weight=1 Count | Train Table Revision / URL | Val Accuracy (%) | Public LB Score | Key Insights / Confusion Shifts |
|---|---|---|---|---|---|---|---|---|
| 0 | 2026-09-05 11:47 | Baseline Seed Only (100/class) | 0 | 600 / 3000 | `/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train` | 69.42% | Submitted | Severe confusion on glacier/mountain/sea and buildings/street. Forest is well-separated. |
| 1 | 2026-09-05 12:07 | Uncertainty Sampling + UMAP Diversity | 800 | 1400 / 3000 | `/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0000` | 78.00% | Pending | +8.58% accuracy leap. Forest reached 94.4% precision. Glacier/mountain (43/17 confusions) remain core bottleneck. |
| 2 | 2026-09-05 12:16 | Error-Focused Sampling (Glacier/Mountain/Urban) | 800 | 2200 / 3000 | `/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0001` | 81.75% | Submitted | +3.75% accuracy leap to 81.75%. Mountain recall jumped to 75.5%, glacier to 69.5%, street F1 to 86.7%. |
| 3 | 2026-09-05 12:38 | Hard-Negative Mining Pass (Boundary Pairs) | 600 | 2800 / 3000 | `/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0002` | 82.00% | Ready | Highest validation accuracy (82.00%). Sea recall 87.5%, forest recall 96.0%, buildings recall 81.0%, glacier precision 85.1%. |

---

### Loop 0: Baseline (Initial Seed Labels Only)
- **Selection Criterion**: No active labeling; strictly the 600 seed labeled images (100 images per class, perfectly balanced).
- **Samples Touched**: 0
- **Cumulative Active Samples (weight=1)**: 600 / 3000
- **Table Revision**: Initial registered table (`train`).
- **Val Accuracy**: 69.42% (best epoch 8/10, step decay scheduler).
- **Kaggle Submission**: `submission.csv` (1,800 rows generated, validated against `sample_submission.csv`).
- **Error Analysis & Confusion Matrix Diagnosis**:
  The baseline ResNet-18 achieves strong discrimination on `forest` (precision 88.2%, recall 89.5%), which forms a clean, distinct cluster in latent feature space. However, significant confusion exists across two critical clusters:
  1. **Glacier vs. Mountain vs. Sea**: 28 true glaciers were predicted as mountains and 39 as sea; 51 true mountains and 31 true seas were cross-misclassified. This stems from visual ambiguity in high-altitude snowy peaks versus glaciated surfaces, as well as water reflections near coastal fjord cliffs.
  2. **Buildings vs. Street**: 42 true buildings were classified as streets (and 26 streets as buildings), reflecting identical architectural textures, asphalt, and urban perspectives in ground-level photography.
  These two specific decision boundaries will form the targeted focus of our upcoming active labeling loops.

---

### Loop 1: Uncertainty / Margin Sampling with UMAP Embedding Diversity
- **Selection Criterion**: Lowest confidence / margin uncertainty from Loop 0 (`nice-vulture`), stratified across 6 predicted classes and dispersed across 3D UMAP clusters via k-means.
- **Samples Touched**: 800 newly curated samples (164 buildings, 102 forest, 103 glacier, 165 mountain, 100 sea, 166 street).
- **Cumulative Active Samples (weight=1)**: 1,400 / 3,000 (enforcing the strict 3,000 budget).
- **Table Revision**: `train_0000` (lineage parent: `train`).
- **Val Accuracy**: 69.42% → **78.00%** (+8.58% absolute gain; best epoch 14/15, cosine warmup + label smoothing 0.1).
- **Kaggle Submission**: `submission.csv` (1,800 test predictions generated and verified; backup in `submissions/submission_20260905_120743.csv`).
- **Confusion Shifts & Key Findings**:
  - `forest` precision jumped to **94.39%** (recall 92.50%), showing nearly perfect boundary definition.
  - `street` recall improved sharply to **89.50%** (up from 76%).
  - `buildings` recall reached **78.50%** (up from 68%), though 33 buildings are still classified as streets.
  - The dominant remaining failure mode is the **Glacier ↔ Mountain ↔ Sea** triad: 43 glaciers predicted as mountain, 17 mountains predicted as glacier, and 32 mountains predicted as sea. Glacier recall remains the lowest at 63.00%, followed by mountain at 66.50%.
  - **Action for Loop 2**: Class-balanced error-focused sampling explicitly oversampling glacier and mountain uncertain points, as well as buildings vs. street boundary cases.

### Loop 2 (2026-09-05 12:16:04)
- **Strategy / Criterion**: Loop 2: Class-Balanced Error-Focused Sampling (Glacier/Mountain/Buildings/Street)
- **Samples Touched / Newly Curated**: 800 samples
- **Total Active Training Samples (weight = 1.0)**: 2200 / 3000
- **Table Revision**: `/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0001` (parent: `/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0000`)
- **Validation Accuracy**: 78.0% → **81.75%** (+3.75%)
- **Confusion Shifts & Key Findings**:
  - Activated 800 curated samples targeting decision boundary uncertainty.
  - Proof artifacts generated: `reports/loop2/confusion_matrix.png`, `confusion_matrix.csv`, `confusion_matrix_report.txt`, `embedding_view.png`.
  - Predictions generated and formatted to `submission.csv` and archived to `submissions/`.

---

### Loop 3: Dedicated Hard-Negative Mining Pass (2026-09-05 12:38:44)
- **Strategy / Criterion**: Hard-Negative Mining pass focused specifically on resolving difficult boundary ambiguous pairs (`glacier` ↔ `mountain` ↔ `sea` and `buildings` ↔ `street`). Candidates identified using prediction margin thresholding and 3D UMAP manifold proximity.
- **Samples Touched / Newly Curated**: 600 samples (bringing total active samples to 2,800, safely within the 3,000 ceiling).
- **Total Active Training Samples (weight = 1.0)**: 2,800 / 3,000 (100% compliant with competition rules).
- **Table Revision**: `/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0002` (lineage parent: `train_0001`).
- **Validation Accuracy**: 81.75% → **82.00%** (best epoch 15/15, lr=1e-6 cosine warmup).
- **Kaggle Submission**: `submission.csv` (1,800 rows; backup in `submissions/submission_20260905_123946.csv`).
- **Confusion Shifts & Detailed Class Performance**:
  - `forest`: **96.00% recall**, 88.48% precision, F1 **0.9209** (best separated class).
  - `sea`: **87.50% recall**, 81.40% precision, F1 **0.8434** (major improvement in coastal water vs. glacier distinction).
  - `street`: **88.00% recall**, 84.21% precision, F1 **0.8606** (urban textures well separated).
  - `buildings`: **81.00% recall**, 77.51% precision, F1 **0.7922** (buildings recall surpassed 80% threshold).
  - `glacier`: 65.50% recall, **85.06% precision**, F1 **0.7401** (false positives reduced dramatically).
  - `mountain`: **74.00% recall**, 75.51% precision, F1 **0.7475**.
- **Proof Artifacts**:
  - Confusion Matrix: `reports/loop3/confusion_matrix.png`
  - Confusion Matrix CSV: `reports/loop3/confusion_matrix.csv`
  - Classification Metrics: `reports/loop3/confusion_matrix_report.txt`
  - 3D UMAP Embedding Visualization: `reports/loop3/embedding_view.png`
