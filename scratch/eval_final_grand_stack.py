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
from scipy.optimize import minimize
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

print("Extracting 2-Scale Multi-Crop predictions for all models...")

p_swa, targets = predict_2scale("scratch/clean_model_seed42_v9_swa.pth")
p_triv, _ = predict_2scale("scratch/clean_model_seed42_v9_trivial.pth")
p_bal1, _ = predict_2scale("scratch/clean_model_seed42_v9_balanced.pth")
p_bal2, _ = predict_2scale("scratch/clean_model_seed43_v9_glacier2x.pth")

ckpts_v9 = [f"scratch/clean_model_seed{s}_v9.pth" for s in [42, 43, 44]]
p_v9 = np.mean([predict_2scale(c)[0] for c in ckpts_v9], axis=0)

ckpts_v5 = [f"scratch/clean_model_seed{s}_v5.pth" for s in [42, 43, 44]]
p_v5 = np.mean([predict_2scale(c)[0] for c in ckpts_v5], axis=0)

models = [p_v5, p_v9, p_bal1, p_bal2, p_triv, p_swa]
names = [
    "v5 3-seed baseline",
    "v9 3-seed standard",
    "v9 balanced seed 42 (1x inv-freq)",
    "v9 balanced seed 43 (2x glacier)",
    "v9 trivial seed 42",
    "v9 SWA seed 42"
]

print("\n" + "=" * 70)
print("STANDALONE 2-SCALE TTA ACCURACIES:")
print("=" * 70)
for name, p in zip(names, models):
    acc = 100.0 * (p.argmax(axis=1) == targets).mean()
    rec = recall_score(targets, p.argmax(axis=1), average=None) * 100
    print(f"  {name:36s}: Acc={acc:.2f}% | Glacier={rec[2]:.1f}% | Mtn={rec[3]:.1f}%")

print("\n" + "=" * 70)
print("6-WAY GRAND STACK (ZERO val-fitting fixed weights):")
print("=" * 70)

policies = {
    "Equal (1/6 each)": [1/6]*6,
    "Principled Prior (0.20 v5, 0.15 v9, 0.15 bal1, 0.15 bal2, 0.15 triv, 0.20 swa)": [0.20, 0.15, 0.15, 0.15, 0.15, 0.20],
    "SWA Priority (0.15 v5, 0.15 v9, 0.15 bal1, 0.15 bal2, 0.15 triv, 0.25 swa)": [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
}

for pname, w in policies.items():
    p_blend = sum(w[i] * models[i] for i in range(6))
    acc = 100.0 * (p_blend.argmax(axis=1) == targets).mean()
    rec = recall_score(targets, p_blend.argmax(axis=1), average=None) * 100
    print(f"\nPolicy: {pname}")
    print(f"  --> Val Acc: {acc:.2f}% | Glacier Rec: {rec[2]:.1f}% | Mtn Rec: {rec[3]:.1f}% | Forest: {rec[1]:.1f}%")

# Strict 5-Fold Cross Validation
print("\n" + "=" * 70)
print("STRICT 5-FOLD CROSS-VALIDATION ON 6-WAY GRAND STACK:")
print("=" * 70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros_like(p_v5)
fold_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(p_v5, targets)):
    y_tr = targets[train_idx]
    def loss_fn(weights):
        w = np.maximum(weights, 0)
        if w.sum() == 0: return 999.0
        w = w / w.sum()
        p_fold = sum(w[i] * models[i][train_idx] for i in range(6))
        eps = 1e-12
        p_fold = np.clip(p_fold, eps, 1.0 - eps)
        return -np.mean(np.log(p_fold[np.arange(len(y_tr)), y_tr]))
        
    res = minimize(loss_fn, [1/6]*6, method='Nelder-Mead')
    best_w = np.maximum(res.x, 0)
    best_w = best_w / best_w.sum()
    
    p_heldout = sum(best_w[i] * models[i][val_idx] for i in range(6))
    oof_preds[val_idx] = p_heldout
    acc_fold = 100.0 * (p_heldout.argmax(axis=1) == targets[val_idx]).mean()
    fold_accs.append(acc_fold)
    print(f"  Fold {fold+1} Held-out Acc: {acc_fold:.2f}%")

oof_acc = 100.0 * (oof_preds.argmax(axis=1) == targets).mean()
print("-" * 70)
print(f"Strict OOF 6-Way Grand Stack Accuracy: {oof_acc:.2f}% (Mean fold: {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%)")
print("=" * 70)
