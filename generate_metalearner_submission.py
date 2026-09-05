import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, load_config
from src.model import ResNet18Classifier
from src.dataset import TestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

set_seed(42)
config = load_config()

# 1. Load saved OOF predictions from scratch/eval_meta_learner.py
oof_preds = np.load("scratch/oof_preds_grand.npy")   # (1200, 6, 6)
y_true = np.load("scratch/oof_targets_grand.npy")      # (1200,)

N, M, C = oof_preds.shape
X_meta = oof_preds.reshape(N, M * C)

# Verify 5-fold CV with C=1.0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
cv_scores = cross_val_score(meta, X_meta, y_true, cv=skf)
print(f"Verified Meta-Learner CV (C=1.0): Mean = {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
print(f"Fold Accuracies: {[round(s*100, 2) for s in cv_scores]}")

# Fit meta-learner on all validation data
meta.fit(X_meta, y_true)
print("[OK] Meta-learner fitted on full validation set.")

# 2. Extract Test Predictions for all 6 models using 2-Scale Multi-Crop TTA
test_dir = PROJECT_ROOT / config["data"]["paths"]["test_dir"]
sample_sub_path = PROJECT_ROOT / config["data"]["paths"]["sample_submission"]

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
tf_zoom = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])

test_dataset_std = TestDataset(test_dir, transform=tf_std, image_size=150)
test_dataset_zoom = TestDataset(test_dir, transform=tf_zoom, image_size=150)

loader_std = DataLoader(test_dataset_std, batch_size=32, shuffle=False, num_workers=2)
loader_zoom = DataLoader(test_dataset_zoom, batch_size=32, shuffle=False, num_workers=2)

def get_test_probs(ckpt_path):
    print(f"  Inferring 2-Scale TTA on test set: {ckpt_path}")
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    probs_std, probs_zoom = [], []
    with torch.no_grad():
        for imgs, _ in loader_std:
            probs_std.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
        for imgs, _ in loader_zoom:
            probs_zoom.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
    p_s = np.concatenate(probs_std, axis=0)
    p_z = np.concatenate(probs_zoom, axis=0)
    return 0.5 * (p_s + p_z)

# 1. v5 ensemble
p_v5_42 = get_test_probs("scratch/clean_model_seed42_v5.pth")
p_v5_43 = get_test_probs("scratch/clean_model_seed43_v5.pth")
p_v5_44 = get_test_probs("scratch/clean_model_seed44_v5.pth")
p_v5 = (p_v5_42 + p_v5_43 + p_v5_44) / 3.0

# 2. v9 standard ensemble
p_v9_42 = get_test_probs("scratch/clean_model_seed42_v9.pth")
p_v9_43 = get_test_probs("scratch/clean_model_seed43_v9.pth")
p_v9_44 = get_test_probs("scratch/clean_model_seed44_v9.pth")
p_v9 = (p_v9_42 + p_v9_43 + p_v9_44) / 3.0

# 3. v9 balanced 1x
p_bal1 = get_test_probs("scratch/clean_model_seed42_v9_balanced.pth")

# 4. v9 balanced 2x
p_bal2 = get_test_probs("scratch/clean_model_seed43_v9_glacier2x.pth")

# 5. v9 trivial
p_triv = get_test_probs("scratch/clean_model_seed42_v9_trivial.pth")

# 6. v9 SWA
p_swa = get_test_probs("scratch/clean_model_seed42_v9_swa.pth")

# Stack in identical order as training: [p_v5, p_v9, p_bal1, p_bal2, p_triv, p_swa]
test_preds = np.stack([p_v5, p_v9, p_bal1, p_bal2, p_triv, p_swa], axis=1) # (1800, 6, 6)
X_test_meta = test_preds.reshape(test_preds.shape[0], M * C)

meta_probs = meta.predict_proba(X_test_meta)
meta_preds = meta_probs.argmax(axis=1)
meta_conf = meta_probs.max(axis=1)

image_ids = [p.stem for p in test_dataset_std.images]
sample_df = pd.read_csv(sample_sub_path)

out_df = pd.DataFrame({
    "image_id": image_ids,
    "prediction": meta_preds,
    "confidence": meta_conf
}).set_index("image_id").reindex(sample_df["image_id"]).reset_index()

# Save meta-learner candidate
meta_sub_path = PROJECT_ROOT / "submission_meta_learner_8142.csv"
out_df.to_csv(meta_sub_path, index=False)
print(f"\n[OK] Saved meta-learner submission to {meta_sub_path.name}")

# Compare with current submission.csv (which got 0.81777 on LB)
cur_df = pd.read_csv(PROJECT_ROOT / "submission.csv")
diff_count = (out_df["prediction"] != cur_df["prediction"]).sum()
pct_agree = 100.0 * (1.0 - diff_count / len(out_df))
print(f"\nComparison with Current Submission (0.81777 LB):")
print(f"  Total differences: {diff_count} / {len(out_df)} ({100.0 * diff_count / len(out_df):.2f}%)")
print(f"  Agreement rate   : {pct_agree:.2f}%")

print("\nMeta-Learner Test Class Distribution:")
for idx, c in enumerate(CLASSES):
    count = (out_df["prediction"] == idx).sum()
    pct = 100.0 * count / len(out_df)
    cur_count = (cur_df["prediction"] == idx).sum()
    print(f"  {c:10s} (class {idx}): {count:4d} ({pct:.1f}%) [was {cur_count:4d}]")
