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

def eval_model_2scale(ckpt_path):
    print(f"\n--- Evaluating {Path(ckpt_path).name} with 2-Scale TTA ---")
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    p_std, p_zoom = [], []
    with torch.no_grad():
        for imgs, _ in loader_std:
            p_std.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
        for imgs, _ in loader_zoom:
            p_zoom.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
    
    p_s = np.concatenate(p_std, axis=0)
    p_z = np.concatenate(p_zoom, axis=0)
    p_2s = 0.5 * (p_s + p_z)
    
    acc_std = 100.0 * (p_s.argmax(1) == targets).mean()
    acc_2s = 100.0 * (p_2s.argmax(1) == targets).mean()
    rec = recall_score(targets, p_2s.argmax(1), average=None) * 100
    
    print(f"  Standard 150x150 Accuracy: {acc_std:.2f}%")
    print(f"  2-Scale TTA Accuracy     : {acc_2s:.2f}%")
    for c, r in zip(CLASSES, rec):
        print(f"    {c:10s}: Recall={r:.1f}%")
    return p_2s, acc_2s

# 1. Focal Loss Model
p_focal, acc_focal = eval_model_2scale("scratch/clean_model_seed42_v9_focal.pth")

# 2. Balanced SWA Model
p_bal_swa, acc_bal_swa = eval_model_2scale("scratch/clean_model_seed42_v9_balanced_swa.pth")

# 3. Rotation TTA on SWA model
print("\n--- Evaluating Rotation TTA on SWA Model ---")
swa_model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
swa_model.load_state_dict(torch.load("scratch/clean_model_seed42_v9_swa.pth", map_location=device))
swa_model.eval()

angles = [-12, -6, 0, 6, 12]
probs_rot = []
with torch.no_grad():
    for img_path, _ in samples:
        img = Image.open(img_path).convert("RGB")
        r_sum = None
        for a in angles:
            rot_img = transforms.functional.rotate(img, a)
            t_rot = tf_std(rot_img).unsqueeze(0).to(device)
            p = F.softmax(swa_model(t_rot), dim=1)
            r_sum = p if r_sum is None else r_sum + p
        probs_rot.append((r_sum / len(angles)).cpu().numpy())

p_rot = np.concatenate(probs_rot, axis=0)
acc_rot = 100.0 * (p_rot.argmax(1) == targets).mean()
print(f"  Rotation TTA Only Accuracy: {acc_rot:.2f}%")

# Load existing 2-scale SWA probs
p_swa_2s, acc_swa_2s = eval_model_2scale("scratch/clean_model_seed42_v9_swa.pth")

# Blends:
p_blend_50 = 0.5 * p_swa_2s + 0.5 * p_rot
acc_blend_50 = 100.0 * (p_blend_50.argmax(1) == targets).mean()

p_blend_70 = 0.7 * p_swa_2s + 0.3 * p_rot
acc_blend_70 = 100.0 * (p_blend_70.argmax(1) == targets).mean()

print(f"  Combined 50/50 (Multi-Crop + Rotation): {acc_blend_50:.2f}%")
print(f"  Combined 70/30 (Multi-Crop + Rotation): {acc_blend_70:.2f}%")
print(f"  (Multi-Crop Alone was {acc_swa_2s:.2f}%)")

print("\n" + "=" * 70)
print("EXPERIMENTS SUMMARY REPORT:")
print("=" * 70)
print(f"  1. Stacked Meta-Learner CV : 81.42% +/- 1.55% (vs 81.25% fixed weights)")
print(f"  2. Focal Loss 2-Scale TTA  : {acc_focal:.2f}%")
print(f"  3. Balanced SWA 2-Scale TTA: {acc_bal_swa:.2f}%")
print(f"  4. Rotation TTA Combined   : {acc_blend_50:.2f}% (50/50), {acc_blend_70:.2f}% (70/30)")
print("=" * 70)
