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

# 1. Load OOF base features & Multi-Res feature
oof_base = np.load("scratch/oof_preds_grand.npy") # (1200, 6, 6)
y_true = np.load("scratch/oof_targets_grand.npy")   # (1200,)
p_176_val = np.load("scratch/val_probs_seed46_176.npy") # (1200, 6)

oof_7way = np.concatenate([oof_base, p_176_val[:, np.newaxis, :]], axis=1) # (1200, 7, 6)
N, M, C = oof_7way.shape
X_meta_7way = oof_7way.reshape(N, M * C)

# Verify CV on 7-way stack
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_7way = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
cv_scores = cross_val_score(meta_7way, X_meta_7way, y_true, cv=skf)
print(f"Verified 7-Way Multi-Res Meta-Learner CV (C=1.0): Mean = {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
print(f"Fold Accuracies: {[round(s*100, 2) for s in cv_scores]}")

meta_7way.fit(X_meta_7way, y_true)
print("[OK] Meta-Learner fitted on full 1,200 validation samples.")

# 2. Extract Test Predictions for Multi-Res 176x176
test_dir = PROJECT_ROOT / config["data"]["paths"]["test_dir"]
sample_sub_path = PROJECT_ROOT / config["data"]["paths"]["sample_submission"]

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_std_176 = transforms.Compose([transforms.Resize((176, 176)), transforms.ToTensor(), norm])
tf_zoom_176 = transforms.Compose([transforms.Resize(188), transforms.CenterCrop(176), transforms.ToTensor(), norm])

test_ds_std_176 = TestDataset(test_dir, transform=tf_std_176, image_size=176)
test_ds_zoom_176 = TestDataset(test_dir, transform=tf_zoom_176, image_size=176)

loader_std_176 = DataLoader(test_ds_std_176, batch_size=32, shuffle=False, num_workers=2)
loader_zoom_176 = DataLoader(test_ds_zoom_176, batch_size=32, shuffle=False, num_workers=2)

print("Inferring Multi-Res 176x176 with 2-Scale TTA on test set...")
model_176 = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
model_176.load_state_dict(torch.load("scratch/clean_model_seed46_v9_176.pth", map_location=device))
model_176.eval()

p_std_176, p_zoom_176 = [], []
with torch.no_grad():
    for imgs, _ in loader_std_176:
        p_std_176.append(F.softmax(model_176(imgs.to(device)), dim=1).cpu().numpy())
    for imgs, _ in loader_zoom_176:
        p_zoom_176.append(F.softmax(model_176(imgs.to(device)), dim=1).cpu().numpy())

p_s_176 = np.concatenate(p_std_176, axis=0)
p_z_176 = np.concatenate(p_zoom_176, axis=0)
p_176_test = 0.5 * (p_s_176 + p_z_176) # (1800, 6)

# Load existing 6 test predictions from generate_metalearner_submission.py
# Or re-extract from previous run
tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
tf_zoom = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])
test_dataset_std = TestDataset(test_dir, transform=tf_std, image_size=150)
test_dataset_zoom = TestDataset(test_dir, transform=tf_zoom, image_size=150)
loader_std = DataLoader(test_dataset_std, batch_size=32, shuffle=False, num_workers=2)
loader_zoom = DataLoader(test_dataset_zoom, batch_size=32, shuffle=False, num_workers=2)

def get_test_probs(ckpt_path):
    m = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.eval()
    p_s, p_z = [], []
    with torch.no_grad():
        for imgs, _ in loader_std: p_s.append(F.softmax(m(imgs.to(device)), dim=1).cpu().numpy())
        for imgs, _ in loader_zoom: p_z.append(F.softmax(m(imgs.to(device)), dim=1).cpu().numpy())
    return 0.5 * (np.concatenate(p_s, axis=0) + np.concatenate(p_z, axis=0))

print("Assembling 7-Way test predictions...")
p_v5 = (get_test_probs("scratch/clean_model_seed42_v5.pth") + get_test_probs("scratch/clean_model_seed43_v5.pth") + get_test_probs("scratch/clean_model_seed44_v5.pth")) / 3.0
p_v9 = (get_test_probs("scratch/clean_model_seed42_v9.pth") + get_test_probs("scratch/clean_model_seed43_v9.pth") + get_test_probs("scratch/clean_model_seed44_v9.pth")) / 3.0
p_bal1 = get_test_probs("scratch/clean_model_seed42_v9_balanced.pth")
p_bal2 = get_test_probs("scratch/clean_model_seed43_v9_glacier2x.pth")
p_triv = get_test_probs("scratch/clean_model_seed42_v9_trivial.pth")
p_swa = get_test_probs("scratch/clean_model_seed42_v9_swa.pth")

test_preds_7way = np.stack([p_v5, p_v9, p_bal1, p_bal2, p_triv, p_swa, p_176_test], axis=1) # (1800, 7, 6)
X_test_7way = test_preds_7way.reshape(test_preds_7way.shape[0], M * C)

meta_probs = meta_7way.predict_proba(X_test_7way)
meta_preds = meta_probs.argmax(axis=1)
meta_conf = meta_probs.max(axis=1)

image_ids = [p.stem for p in test_dataset_std.images]
sample_df = pd.read_csv(sample_sub_path)

out_df = pd.DataFrame({
    "image_id": image_ids,
    "prediction": meta_preds,
    "confidence": meta_conf
}).set_index("image_id").reindex(sample_df["image_id"]).reset_index()

out_path = PROJECT_ROOT / "submission_7way_multires_8175.csv"
out_df.to_csv(out_path, index=False)
print(f"\n[OK] Saved 7-Way Multi-Res submission to {out_path.name}")

# Comparison with current 0.82888 submission
cur_df = pd.read_csv(PROJECT_ROOT / "submission_meta_learner_8142.csv")
diff_count = (out_df["prediction"] != cur_df["prediction"]).sum()
pct_agree = 100.0 * (1.0 - diff_count / len(out_df))

print(f"\nComparison with Current 0.82888 Submission:")
print(f"  Total differences: {diff_count} / {len(out_df)} ({100.0 * diff_count / len(out_df):.2f}%)")
print(f"  Agreement rate   : {pct_agree:.2f}%")

print("\n7-Way Multi-Res Test Class Distribution:")
for idx, c in enumerate(CLASSES):
    count = (out_df["prediction"] == idx).sum()
    pct = 100.0 * count / len(out_df)
    cur_count = (cur_df["prediction"] == idx).sum()
    print(f"  {c:10s} (class {idx}): {count:4d} ({pct:.1f}%) [was {cur_count:4d}]")
