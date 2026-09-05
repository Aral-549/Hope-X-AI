import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ResNet18Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

val_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()
tf_base = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ValDataset(Dataset):
    def __init__(self, table, transform):
        self.samples = [(table[i]["image"], table[i]["label"]) for i in range(len(table))]
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.transform(Image.open(self.samples[idx][0]).convert("RGB")), self.samples[idx][1]

loader = DataLoader(ValDataset(val_table, tf_base), batch_size=32, shuffle=False, num_workers=2)

def predict_single(ckpt_path):
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    probs, targets = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            p = F.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            probs.append(p)
            targets.extend(lbls.numpy())
    return np.concatenate(probs, axis=0), np.array(targets)

print("Extracting clean forward passes (NO flip)...")
p_swa, targets = predict_single("scratch/clean_model_seed42_v9_swa.pth")
p_triv, _ = predict_single("scratch/clean_model_seed42_v9_trivial.pth")
p_bal, _ = predict_single("scratch/clean_model_seed42_v9_balanced.pth")

ckpts_v9 = [f"scratch/clean_model_seed{s}_v9.pth" for s in [42, 43, 44]]
p_v9 = np.mean([predict_single(c)[0] for c in ckpts_v9], axis=0)

ckpts_v5 = [f"scratch/clean_model_seed{s}_v5.pth" for s in [42, 43, 44]]
p_v5 = np.mean([predict_single(c)[0] for c in ckpts_v5], axis=0)

print("\n" + "=" * 70)
print("STANDALONE MODEL VALIDATION ACCURACIES (Standard 150x150, No Flip):")
print("=" * 70)
for name, p in [("v5 3-seed ensemble", p_v5),
                ("v9 standard 3-seed ensemble", p_v9),
                ("v9 balanced seed 42", p_bal),
                ("v9 trivial seed 42", p_triv),
                ("v9 SWA seed 42", p_swa)]:
    acc = 100.0 * (p.argmax(axis=1) == targets).mean()
    print(f"  {name:32s} : {acc:.2f}%")

print("\n" + "=" * 70)
print("5-WAY STACK WITH FIXED / PRINCIPLED WEIGHTS (ZERO val-fitting):")
print("=" * 70)

policies = {
    "Equal (0.20 each across all 5)": [0.20, 0.20, 0.20, 0.20, 0.20],
    "SWA Priority (0.20 v5, 0.15 v9, 0.15 bal, 0.20 triv, 0.30 swa)": [0.20, 0.15, 0.15, 0.20, 0.30],
    "Balanced Prior (0.25 v5, 0.25 v9, 0.15 bal, 0.15 triv, 0.20 swa)": [0.25, 0.25, 0.15, 0.15, 0.20],
    "Conservative (0.30 v5, 0.20 v9, 0.10 bal, 0.15 triv, 0.25 swa)": [0.30, 0.20, 0.10, 0.15, 0.25],
}

models = [p_v5, p_v9, p_bal, p_triv, p_swa]

for name, w in policies.items():
    p_blend = sum(w[i] * models[i] for i in range(5))
    acc = 100.0 * (p_blend.argmax(axis=1) == targets).mean()
    rec = recall_score(targets, p_blend.argmax(axis=1), average=None) * 100
    print(f"\nPolicy: {name}")
    print(f"  --> Val Acc: {acc:.2f}% | Glacier Rec: {rec[2]:.1f}% | Mtn Rec: {rec[3]:.1f}% | Forest: {rec[1]:.1f}%")

# Strict 5-fold CV for the 5-way stack
print("\n" + "=" * 70)
print("STRICT 5-FOLD CROSS-VALIDATION ON 5-WAY STACK:")
print("=" * 70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
from scipy.optimize import minimize

oof_preds = np.zeros_like(p_v5)
fold_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(p_v5, targets)):
    y_tr = targets[train_idx]
    def loss_fn(weights):
        w = np.maximum(weights, 0)
        if w.sum() == 0: return 999.0
        w = w / w.sum()
        p_fold = sum(w[i] * models[i][train_idx] for i in range(5))
        eps = 1e-12
        p_fold = np.clip(p_fold, eps, 1.0 - eps)
        return -np.mean(np.log(p_fold[np.arange(len(y_tr)), y_tr]))
        
    res = minimize(loss_fn, [0.20]*5, method='Nelder-Mead')
    best_w = np.maximum(res.x, 0)
    best_w = best_w / best_w.sum()
    
    p_heldout = sum(best_w[i] * models[i][val_idx] for i in range(5))
    oof_preds[val_idx] = p_heldout
    acc_fold = 100.0 * (p_heldout.argmax(axis=1) == targets[val_idx]).mean()
    fold_accs.append(acc_fold)

oof_acc = 100.0 * (oof_preds.argmax(axis=1) == targets).mean()
print(f"Strict OOF 5-Way Stack Accuracy: {oof_acc:.2f}% (Mean fold: {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%)")
print("=" * 70)
