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

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class RawValDataset(Dataset):
    def __init__(self, table):
        self.samples = [(table[i]["image"], table[i]["label"]) for i in range(len(table))]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return Image.open(self.samples[idx][0]).convert("RGB"), self.samples[idx][1]

dataset = RawValDataset(val_table)

# Load SWA model
model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
model.load_state_dict(torch.load("scratch/clean_model_seed42_v9_swa.pth", map_location=device))
model.eval()

# Transforms:
# 1. Standard: 150x150
tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
# 2. Slight Zoom 160 -> CenterCrop 150
tf_zoom1 = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])
# 3. Slight Zoom 170 -> CenterCrop 150
tf_zoom2 = transforms.Compose([transforms.Resize(170), transforms.CenterCrop(150), transforms.ToTensor(), norm])

targets = [dataset[i][1] for i in range(len(dataset))]
targets = np.array(targets)

probs_std = []
probs_zoom1 = []
probs_zoom2 = []

print("Running multi-crop TTA evaluation (NO horizontal flips)...")
with torch.no_grad():
    for i in range(len(dataset)):
        img, _ = dataset[i]
        
        t1 = tf_std(img).unsqueeze(0).to(device)
        p1 = F.softmax(model(t1), dim=1).cpu().numpy()
        probs_std.append(p1)
        
        t2 = tf_zoom1(img).unsqueeze(0).to(device)
        p2 = F.softmax(model(t2), dim=1).cpu().numpy()
        probs_zoom1.append(p2)
        
        t3 = tf_zoom2(img).unsqueeze(0).to(device)
        p3 = F.softmax(model(t3), dim=1).cpu().numpy()
        probs_zoom2.append(p3)

p_std = np.concatenate(probs_std, axis=0)
p_z1 = np.concatenate(probs_zoom1, axis=0)
p_z2 = np.concatenate(probs_zoom2, axis=0)

# Evaluate
acc_std = 100.0 * (p_std.argmax(axis=1) == targets).mean()
acc_z1 = 100.0 * (p_z1.argmax(axis=1) == targets).mean()
acc_z2 = 100.0 * (p_z2.argmax(axis=1) == targets).mean()

# 2-Scale TTA (std + zoom1)
p_2scale = 0.5 * (p_std + p_z1)
acc_2scale = 100.0 * (p_2scale.argmax(axis=1) == targets).mean()

# 3-Scale TTA (0.5 std + 0.25 z1 + 0.25 z2)
p_3scale = 0.5 * p_std + 0.25 * p_z1 + 0.25 * p_z2
acc_3scale = 100.0 * (p_3scale.argmax(axis=1) == targets).mean()

print("=" * 70)
print(f"SWA SINGLE-MODEL MULTI-CROP / MULTI-SCALE TTA RESULTS (NO FLIP):")
print("=" * 70)
print(f"  Standard 150x150                     : {acc_std:.2f}%")
print(f"  Zoom 1 (160 -> CenterCrop 150)       : {acc_z1:.2f}%")
print(f"  Zoom 2 (170 -> CenterCrop 150)       : {acc_z2:.2f}%")
print(f"  2-Scale TTA (0.5 std + 0.5 zoom1)    : {acc_2scale:.2f}%")
print(f"  3-Scale TTA (0.5 std + 0.25 z1 + 0.25 z2): {acc_3scale:.2f}%")
print("=" * 70)
