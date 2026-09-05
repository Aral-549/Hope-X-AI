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

def extract_logits(ckpt_path):
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    logits_std, logits_zoom, targets = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader_std:
            logits_std.append(model(imgs.to(device)).cpu().numpy())
            targets.extend(lbls.numpy())
        for imgs, _ in loader_zoom:
            logits_zoom.append(model(imgs.to(device)).cpu().numpy())
            
    l_s = np.concatenate(logits_std, axis=0)
    l_z = np.concatenate(logits_zoom, axis=0)
    return 0.5 * (l_s + l_z), np.array(targets)

print("Loading cached logits...")
l_v5_42, targets = extract_logits("scratch/clean_model_seed42_v5.pth")
l_v5_43, _ = extract_logits("scratch/clean_model_seed43_v5.pth")
l_v5_44, _ = extract_logits("scratch/clean_model_seed44_v5.pth")
l_v5_ens = (l_v5_42 + l_v5_43 + l_v5_44) / 3.0

l_v9_42, _ = extract_logits("scratch/clean_model_seed42_v9.pth")
l_v9_43, _ = extract_logits("scratch/clean_model_seed43_v9.pth")
l_v9_44, _ = extract_logits("scratch/clean_model_seed44_v9.pth")
l_v9_ens = (l_v9_42 + l_v9_43 + l_v9_44) / 3.0

l_bal1, _ = extract_logits("scratch/clean_model_seed42_v9_balanced.pth")
l_bal2_g2x, _ = extract_logits("scratch/clean_model_seed43_v9_glacier2x.pth")
l_triv, _ = extract_logits("scratch/clean_model_seed42_v9_trivial.pth")
l_swa, _ = extract_logits("scratch/clean_model_seed42_v9_swa.pth")

def fit_temperature(l, y):
    def nll(t):
        scaled = l / max(t[0], 0.01)
        return F.cross_entropy(torch.tensor(scaled), torch.tensor(y)).item()
    res = minimize(nll, [1.0], method='Nelder-Mead')
    return max(res.x[0], 0.05)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_stack(comp_logits, comp_names, label=""):
    print("\n" + "=" * 70)
    print(f"EVALUATING STACK: {label}")
    print("=" * 70)
    n_comp = len(comp_logits)
    oof_uncalib = np.zeros((len(targets), 6))
    oof_calib = np.zeros((len(targets), 6))
    oof_opt = np.zeros((len(targets), 6))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(comp_logits[0], targets)):
        y_tr = targets[train_idx]
        y_val = targets[val_idx]
        
        # 1. Uncalibrated equal weights
        p_uncal = np.mean([F.softmax(torch.tensor(cl[val_idx]), dim=1).numpy() for cl in comp_logits], axis=0)
        oof_uncalib[val_idx] = p_uncal
        
        # 2. Fit temps on train_idx
        temps = [fit_temperature(cl[train_idx], y_tr) for cl in comp_logits]
        calib_tr = [F.softmax(torch.tensor(cl[train_idx] / temps[i]), dim=1).numpy() for i, cl in enumerate(comp_logits)]
        calib_val = [F.softmax(torch.tensor(cl[val_idx] / temps[i]), dim=1).numpy() for i, cl in enumerate(comp_logits)]
        
        oof_calib[val_idx] = np.mean(calib_val, axis=0)
        
        # 3. Fit weights on train_idx
        def loss_w(w):
            w = np.maximum(w, 0)
            if w.sum() == 0: return 999.0
            w = w / w.sum()
            p = sum(w[i] * calib_tr[i] for i in range(n_comp))
            eps = 1e-12
            p = np.clip(p, eps, 1.0 - eps)
            return -np.mean(np.log(p[np.arange(len(y_tr)), y_tr]))
        res_w = minimize(loss_w, [1.0/n_comp]*n_comp, method='Nelder-Mead')
        best_w = np.maximum(res_w.x, 0)
        best_w = best_w / best_w.sum()
        
        oof_opt[val_idx] = sum(best_w[i] * calib_val[i] for i in range(n_comp))
        
    acc_un = 100.0 * (oof_uncalib.argmax(axis=1) == targets).mean()
    acc_cal = 100.0 * (oof_calib.argmax(axis=1) == targets).mean()
    acc_opt = 100.0 * (oof_opt.argmax(axis=1) == targets).mean()
    
    rec_opt = recall_score(targets, oof_opt.argmax(axis=1), average=None) * 100
    print(f"1. OOF Uncalibrated Equal: {acc_un:.2f}%")
    print(f"2. OOF Calibrated Equal  : {acc_cal:.2f}%")
    print(f"3. OOF Calibrated Optimal: {acc_opt:.2f}%")
    print("Per-class recall (Calibrated Optimal):")
    for c, r in zip(CLASSES, rec_opt):
        print(f"  {c:10s}: {r:.1f}%")
    return acc_opt, oof_opt

# Option A: 5-way stack WITHOUT glacier 2x (keeping high mountain recall)
comps_5 = [l_v5_ens, l_v9_ens, l_bal1, l_triv, l_swa]
names_5 = ["v5_ens", "v9_ens", "v9_bal1", "v9_triv", "v9_swa"]
evaluate_stack(comps_5, names_5, label="5-Way Clean Stack (NO glacier 2x)")

# Option B: 6-way stack WITH glacier 2x
comps_6 = [l_v5_ens, l_v9_ens, l_bal1, l_triv, l_swa, l_bal2_g2x]
names_6 = ["v5_ens", "v9_ens", "v9_bal1", "v9_triv", "v9_swa", "v9_bal2_g2x"]
evaluate_stack(comps_6, names_6, label="6-Way Grand Stack (WITH glacier 2x)")
