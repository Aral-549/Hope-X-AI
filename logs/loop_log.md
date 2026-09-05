# HackBlox 2026 · 3LC Data-Centric AI Challenge
## Iterative Labeling & Training Experiment Log

| Loop | Timestamp | Selection Criterion | Samples Touched | Cum. Weight=1 Count | Train Table Revision / URL | Val Accuracy (%) | Public LB Score | Key Insights / Confusion Shifts |
|---|---|---|---|---|---|---|---|---|
| 0 | 2026-09-05 11:47 | Baseline Seed Only (100/class) | 0 | 600 / 3000 | `/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train` | 69.42% | Submitted | Severe confusion on glacier/mountain/sea and buildings/street. Forest is well-separated. |

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
