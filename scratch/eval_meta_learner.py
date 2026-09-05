import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ResNet18Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

val_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
tf_zoom = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])

class ValDataset(Dataset):
    def __init__(self, table, transform):
        self.samples = [(table[i]["image"], table[i]["label"]) for i in range(len(table))]
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.transform(Image.open(self.samples[idx][0]).convert("RGB")), self.samples[idx][1]

loader_std = DataLoader(ValDataset(val_table, tf_std), batch_size=32, shuffle=False, num_workers=2)
loader_zoom = DataLoader(ValDataset(val_table, tf_zoom), batch_size=32, shuffle=False, num_workers=2)

def predict_2scale(ckpt_path):
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    probs_std, probs_zoom, targets = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader_std:
            probs_std.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
            targets.extend(lbls.numpy())
        for imgs, _ in loader_zoom:
            probs_zoom.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
            
    p_s = np.concatenate(probs_std, axis=0)
    p_z = np.concatenate(probs_zoom, axis=0)
    return 0.5 * (p_s + p_z), np.array(targets)

print("Extracting 2-Scale Multi-Crop predictions for all 6 grand stack models...")

p_swa, targets = predict_2scale("scratch/clean_model_seed42_v9_swa.pth")
p_triv, _ = predict_2scale("scratch/clean_model_seed42_v9_trivial.pth")
p_bal1, _ = predict_2scale("scratch/clean_model_seed42_v9_balanced.pth")
p_bal2, _ = predict_2scale("scratch/clean_model_seed43_v9_glacier2x.pth")

ckpts_v9 = [f"scratch/clean_model_seed{s}_v9.pth" for s in [42, 43, 44]]
p_v9 = np.mean([predict_2scale(c)[0] for c in ckpts_v9], axis=0)

ckpts_v5 = [f"scratch/clean_model_seed{s}_v5.pth" for s in [42, 43, 44]]
p_v5 = np.mean([predict_2scale(c)[0] for c in ckpts_v5], axis=0)

# Shape: (N, n_models, 6)
oof_preds = np.stack([p_v5, p_v9, p_bal1, p_bal2, p_triv, p_swa], axis=1)
y_true = targets

N, M, C = oof_preds.shape
X_meta = oof_preds.reshape(N, M * C)

print(f"X_meta shape: {X_meta.shape}, y_true shape: {y_true.shape}")

# Test multiple C regularizations
for c_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    meta = LogisticRegression(max_iter=2000, C=c_val, solver="lbfgs", random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(meta, X_meta, y_true, cv=skf)
    print(f"Meta-learner CV (C={c_val:5.2f}): Mean = {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}% (Folds: {[round(s*100, 2) for s in scores]})")

# Save arrays for reuse
np.save("scratch/oof_preds_grand.npy", oof_preds)
np.save("scratch/oof_targets_grand.npy", y_true)
print("[OK] Saved oof_preds_grand.npy and oof_targets_grand.npy")
