import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, recall_score
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ResNet18Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

val_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()
tf = transforms.Compose([
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

loader = DataLoader(ValDataset(val_table, tf), batch_size=32, shuffle=False, num_workers=2)

def eval_ckpts(ckpt_list):
    models = [ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device) for _ in ckpt_list]
    for m, p in zip(models, ckpt_list):
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
    all_probs = [[] for _ in models]
    targets = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            targets.extend(lbls.numpy())
            for m_idx, m in enumerate(models):
                all_probs[m_idx].append(F.softmax(m(imgs), dim=1).cpu().numpy())
    probs_cat = [np.concatenate(p, axis=0) for p in all_probs]
    ens_p = np.mean(probs_cat, axis=0)
    return ens_p, np.array(targets)

# 1. Evaluate single balanced model
p_bal, targets = eval_ckpts(["scratch/clean_model_seed42_v9_balanced.pth"])
acc_bal = 100.0 * (p_bal.argmax(axis=1) == targets).mean()
rec_bal = recall_score(targets, p_bal.argmax(axis=1), average=None) * 100

# 2. Evaluate train_0009 standard 3-seed ensemble
ckpts_v9 = [
    "scratch/clean_model_seed42_v9.pth",
    "scratch/clean_model_seed43_v9.pth",
    "scratch/clean_model_seed44_v9.pth",
]
p_v9, _ = eval_ckpts(ckpts_v9)
acc_v9 = 100.0 * (p_v9.argmax(axis=1) == targets).mean()
rec_v9 = recall_score(targets, p_v9.argmax(axis=1), average=None) * 100

# 3. Evaluate train_0005 standard 3-seed ensemble
ckpts_v5 = [
    "scratch/clean_model_seed42_v5.pth",
    "scratch/clean_model_seed43_v5.pth",
    "scratch/clean_model_seed44_v5.pth",
]
p_v5, _ = eval_ckpts(ckpts_v5)
acc_v5 = 100.0 * (p_v5.argmax(axis=1) == targets).mean()
rec_v5 = recall_score(targets, p_v5.argmax(axis=1), average=None) * 100

df_rec = pd.DataFrame({
    "Class": CLASSES,
    "train_0009 Standard": rec_v9.round(1),
    "train_0009 Balanced (Single)": rec_bal.round(1),
    "Glacier/Mtn Delta": (rec_bal - rec_v9).round(1),
})

print("=" * 70)
print("  STEP 1: Balanced Sampler vs Standard Sampler Recall on train_0009")
print("=" * 70)
print(df_rec.to_string(index=False))
print("-" * 70)
print(f"Accuracy: train_0009 Standard (3-seed): {acc_v9:.2f}% | Balanced (Single): {acc_bal:.2f}%")
print("=" * 70)

# 4. Test Stacking Blend: Standard v9 + Balanced v9
print("\n--- Testing Stacking Blends (v9 Standard + v9 Balanced) ---")
for w_bal in [0.15, 0.20, 0.25, 0.30]:
    p_stack = (1.0 - w_bal) * p_v9 + w_bal * p_bal
    acc_stack = 100.0 * (p_stack.argmax(axis=1) == targets).mean()
    rec_stack = recall_score(targets, p_stack.argmax(axis=1), average=None) * 100
    print(f"Weight Balanced={w_bal:.2f}: Acc={acc_stack:.2f}% | Glacier Rec={rec_stack[2]:.1f}% | Mountain Rec={rec_stack[3]:.1f}%")

# 5. Test 3-Way Grand Blend: train_0005 (0.35) + train_0009 Standard (0.45) + train_0009 Balanced (0.20)
p_grand = 0.35 * p_v5 + 0.45 * p_v9 + 0.20 * p_bal
acc_grand = 100.0 * (p_grand.argmax(axis=1) == targets).mean()
rec_grand = recall_score(targets, p_grand.argmax(axis=1), average=None) * 100
print("\n" + "=" * 70)
print(f"  3-WAY CLEAN STACK (v5 base + v9 standard + v9 balanced): {acc_grand:.2f}%")
print("=" * 70)
for idx, c in enumerate(CLASSES):
    print(f"  {c:10s}: Recall={rec_grand[idx]:.1f}%")
print("=" * 70)
