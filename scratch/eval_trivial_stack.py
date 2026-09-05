import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import recall_score, classification_report
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

def predict_model(ckpt_path, use_tta=True):
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    probs = []
    targets = []
    with torch.no_grad():
        for imgs, lbls in loader_base:
            imgs = imgs.to(device)
            p = F.softmax(model(imgs), dim=1).cpu().numpy()
            probs.append(p)
            targets.extend(lbls.numpy())
    p_base = np.concatenate(probs, axis=0)
    
    if not use_tta:
        return p_base, np.array(targets)
        
    probs_flip = []
    with torch.no_grad():
        for imgs, _ in loader_flip:
            imgs = imgs.to(device)
            p = F.softmax(model(imgs), dim=1).cpu().numpy()
            probs_flip.append(p)
    p_flip = np.concatenate(probs_flip, axis=0)
    
    return 0.5 * (p_base + p_flip), np.array(targets)

print("Loading predictions for all clean models...")

p_triv, targets = predict_model("scratch/clean_model_seed42_v9_trivial.pth", use_tta=True)
p_triv_notta, _ = predict_model("scratch/clean_model_seed42_v9_trivial.pth", use_tta=False)
p_bal, _ = predict_model("scratch/clean_model_seed42_v9_balanced.pth", use_tta=True)

ckpts_v9 = [
    "scratch/clean_model_seed42_v9.pth",
    "scratch/clean_model_seed43_v9.pth",
    "scratch/clean_model_seed44_v9.pth",
]
p_v9_list = [predict_model(c, use_tta=True)[0] for c in ckpts_v9]
p_v9_ens = np.mean(p_v9_list, axis=0)

ckpts_v5 = [
    "scratch/clean_model_seed42_v5.pth",
    "scratch/clean_model_seed43_v5.pth",
    "scratch/clean_model_seed44_v5.pth",
]
p_v5_list = [predict_model(c, use_tta=True)[0] for c in ckpts_v5]
p_v5_ens = np.mean(p_v5_list, axis=0)

acc_triv_notta = 100.0 * (p_triv_notta.argmax(axis=1) == targets).mean()
acc_triv = 100.0 * (p_triv.argmax(axis=1) == targets).mean()
rec_triv = recall_score(targets, p_triv.argmax(axis=1), average=None) * 100

acc_bal = 100.0 * (p_bal.argmax(axis=1) == targets).mean()
rec_bal = recall_score(targets, p_bal.argmax(axis=1), average=None) * 100

acc_v9 = 100.0 * (p_v9_ens.argmax(axis=1) == targets).mean()
acc_v5 = 100.0 * (p_v5_ens.argmax(axis=1) == targets).mean()

print("\n" + "=" * 70)
print(f"SINGLE MODEL RESULTS (train_0009):")
print(f"  TrivialAugment Seed 42 (No TTA): {acc_triv_notta:.2f}%")
print(f"  TrivialAugment Seed 42 (With TTA): {acc_triv:.2f}%")
print(f"  Balanced Sampler Seed 42 (With TTA): {acc_bal:.2f}%")
print(f"  v9 Standard 3-seed Ensemble: {acc_v9:.2f}%")
print(f"  v5 Standard 3-seed Ensemble: {acc_v5:.2f}%")
print("=" * 70)
print("Per-class recall for TrivialAugment:")
for c, r in zip(CLASSES, rec_triv):
    print(f"  {c:10s}: {r:.1f}%")
print("=" * 70)

# Grid search 4-way stack
best_acc = 0.0
best_weights = None
best_preds = None

for w5 in np.linspace(0.15, 0.40, 6):
    for w9 in np.linspace(0.20, 0.50, 7):
        for wbal in np.linspace(0.10, 0.35, 6):
            for wtriv in np.linspace(0.10, 0.35, 6):
                total = w5 + w9 + wbal + wtriv
                p_blend = (w5*p_v5_ens + w9*p_v9_ens + wbal*p_bal + wtriv*p_triv) / total
                acc = 100.0 * (p_blend.argmax(axis=1) == targets).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_weights = (w5/total, w9/total, wbal/total, wtriv/total)
                    best_preds = p_blend

print("\n" + "*" * 70)
print(f"BEST 4-WAY CLEAN STACK VALIDATION ACCURACY: {best_acc:.2f}%")
print(f"Weights (v5, v9_std, v9_bal, v9_triv): {[round(x, 3) for x in best_weights]}")
print("*" * 70)

rec_best = recall_score(targets, best_preds.argmax(axis=1), average=None) * 100
for c, r in zip(CLASSES, rec_best):
    print(f"  {c:10s}: Recall={r:.1f}%")
print("*" * 70)
