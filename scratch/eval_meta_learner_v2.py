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
samples = [(val_table[i]["image"], val_table[i]["label"]) for i in range(len(val_table))]
targets = np.array([s[1] for s in samples])

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
tf_zoom = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])

class EvalDataset(Dataset):
    def __init__(self, samples, tf):
        self.samples = samples
        self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx][0]).convert("RGB")
        return self.tf(img), self.samples[idx][1]

loader_std = DataLoader(EvalDataset(samples, tf_std), batch_size=32, shuffle=False, num_workers=2)
loader_zoom = DataLoader(EvalDataset(samples, tf_zoom), batch_size=32, shuffle=False, num_workers=2)

def predict_2scale(ckpt_path):
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

print("Extracting predictions for candidate models...")
p_swa1 = predict_2scale("scratch/clean_model_seed42_v9_swa.pth")
p_bal_swa = predict_2scale("scratch/clean_model_seed42_v9_balanced_swa.pth")
p_triv = predict_2scale("scratch/clean_model_seed42_v9_trivial.pth")
p_bal2 = predict_2scale("scratch/clean_model_seed43_v9_glacier2x.pth")

ckpts_v9 = [f"scratch/clean_model_seed{s}_v9.pth" for s in [42, 43, 44]]
p_v9 = np.mean([predict_2scale(c) for c in ckpts_v9], axis=0)

ckpts_v5 = [f"scratch/clean_model_seed{s}_v5.pth" for s in [42, 43, 44]]
p_v5 = np.mean([predict_2scale(c) for c in ckpts_v5], axis=0)

# Build stack feature matrix
models_list = [p_v5, p_v9, p_bal2, p_triv, p_swa1, p_bal_swa]
oof_preds = np.stack(models_list, axis=1) # (N, M, 6)
y_true = targets

N, M, C = oof_preds.shape
X_meta = oof_preds.reshape(N, M * C)

print(f"X_meta shape: {X_meta.shape}, y_true shape: {y_true.shape}")

for c_val in [0.1, 0.5, 1.0, 2.0]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta = LogisticRegression(max_iter=2000, C=c_val, solver="lbfgs", random_state=42)
    scores = cross_val_score(meta, X_meta, y_true, cv=skf)
    print(f"Meta-Learner (C={c_val:4.2f}): Mean = {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}% (Folds: {[round(s*100, 2) for s in scores]})")

# Also compare with simple weighted average
p_blend = (0.25 * p_swa1 + 0.25 * p_bal_swa + 0.15 * p_bal2 + 0.15 * p_triv + 0.10 * p_v9 + 0.10 * p_v5)
acc_blend = 100.0 * (p_blend.argmax(1) == y_true).mean()
print(f"\nSimple Equalized-SWA Weighted Blend: {acc_blend:.2f}%")
