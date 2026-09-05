import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
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
tf_flip = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ValDataset(Dataset):
    def __init__(self, table, transform):
        self.samples = [(table[i]["image"], table[i]["label"]) for i in range(len(table))]
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.transform(Image.open(self.samples[idx][0]).convert("RGB")), self.samples[idx][1]

loader_base = DataLoader(ValDataset(val_table, tf_base), batch_size=32, shuffle=False, num_workers=2)
loader_flip = DataLoader(ValDataset(val_table, tf_flip), batch_size=32, shuffle=False, num_workers=2)

def predict_model(ckpt_path):
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    probs_base, probs_flip, targets = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader_base:
            imgs = imgs.to(device)
            probs_base.append(F.softmax(model(imgs), dim=1).cpu().numpy())
            targets.extend(lbls.numpy())
        for imgs, _ in loader_flip:
            imgs = imgs.to(device)
            probs_flip.append(F.softmax(model(imgs), dim=1).cpu().numpy())
            
    p_base = np.concatenate(probs_base, axis=0)
    p_flip = np.concatenate(probs_flip, axis=0)
    return 0.5 * (p_base + p_flip), np.array(targets)

print("Extracting validation predictions for all components...")
p_triv, targets = predict_model("scratch/clean_model_seed42_v9_trivial.pth")
p_bal, _ = predict_model("scratch/clean_model_seed42_v9_balanced.pth")

ckpts_v9 = [f"scratch/clean_model_seed{s}_v9.pth" for s in [42, 43, 44]]
p_v9 = np.mean([predict_model(c)[0] for c in ckpts_v9], axis=0)

ckpts_v5 = [f"scratch/clean_model_seed{s}_v5.pth" for s in [42, 43, 44]]
p_v5 = np.mean([predict_model(c)[0] for c in ckpts_v5], axis=0)

# 1. Test Fixed Principled (Equal or Intuitive) Weights (NO OPTIMIZATION)
print("\n" + "=" * 70)
print("1. PRINCIPLED / FIXED WEIGHTS (Zero val-fitting):")
print("=" * 70)

weights_options = {
    "Equal Weights (0.25, 0.25, 0.25, 0.25)": [0.25, 0.25, 0.25, 0.25],
    "Conservative (0.40 v5, 0.30 v9, 0.15 bal, 0.15 triv)": [0.40, 0.30, 0.15, 0.15],
    "Balanced (0.30 v5, 0.30 v9, 0.20 bal, 0.20 triv)": [0.30, 0.30, 0.20, 0.20],
    "Original Search (0.368, 0.263, 0.105, 0.263)": [0.368, 0.263, 0.105, 0.263]
}

for name, w in weights_options.items():
    p_fixed = w[0]*p_v5 + w[1]*p_v9 + w[2]*p_bal + w[3]*p_triv
    acc = 100.0 * (p_fixed.argmax(axis=1) == targets).mean()
    print(f"  {name:50s} --> Val Acc: {acc:.2f}%")

# 2. Strict 5-Fold Cross-Validation for Weight Optimization
print("\n" + "=" * 70)
print("2. STRICT 5-FOLD CROSS-VALIDATION (Out-of-Fold Evaluation):")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros_like(p_v5)
fold_accs = []
fold_weights = []

models_probs = [p_v5, p_v9, p_bal, p_triv]

for fold, (train_idx, val_idx) in enumerate(skf.split(p_v5, targets)):
    # Optimize weights ONLY on train_idx (4 folds = 960 samples)
    y_tr = targets[train_idx]
    
    def loss_fn(weights):
        w = np.maximum(weights, 0)
        if w.sum() == 0: return 999.0
        w = w / w.sum()
        p_fold = sum(w[i] * models_probs[i][train_idx] for i in range(4))
        # Negative log-likelihood
        eps = 1e-12
        p_fold = np.clip(p_fold, eps, 1.0 - eps)
        nll = -np.mean(np.log(p_fold[np.arange(len(y_tr)), y_tr]))
        return nll
        
    res = minimize(loss_fn, [0.25, 0.25, 0.25, 0.25], method='Nelder-Mead')
    best_w = np.maximum(res.x, 0)
    best_w = best_w / best_w.sum()
    fold_weights.append(best_w)
    
    # Evaluate ONLY on val_idx (1 fold = 240 held-out samples)
    p_heldout = sum(best_w[i] * models_probs[i][val_idx] for i in range(4))
    oof_preds[val_idx] = p_heldout
    acc_fold = 100.0 * (p_heldout.argmax(axis=1) == targets[val_idx]).mean()
    fold_accs.append(acc_fold)
    print(f"  Fold {fold+1} Held-out Acc: {acc_fold:.2f}% | Weights: {[round(x, 3) for x in best_w]}")

oof_overall_acc = 100.0 * (oof_preds.argmax(axis=1) == targets).mean()
print("-" * 70)
print(f"STRICT OUT-OF-FOLD (OOF) CROSS-VALIDATED ACCURACY: {oof_overall_acc:.2f}%")
print(f"Mean Fold Accuracy: {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%")
print("=" * 70)
