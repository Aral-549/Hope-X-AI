import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import recall_score
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, load_config
from src.model import ResNet18Classifier
from src.dataset import TableSampleMapper
from src.augment import get_advanced_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

set_seed(42)
config = load_config()

# 1. Load Tables
train_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="train_0009").latest()
val_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()

# Compute class weights for inverse-frequency sampling BEFORE mapping
active_rows = [row for row in train_table.table_rows if row["weight"] > 0]
class_counts = pd.Series([row["label"] for row in active_rows]).value_counts().to_dict()

sample_weights = []
for row in train_table.table_rows:
    if row["weight"] > 0:
        w = 1.0 / max(1, class_counts[row["label"]])
        sample_weights.append(w)
    else:
        sample_weights.append(0.0)

sample_weights = torch.tensor(sample_weights, dtype=torch.double)
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(active_rows),
    replacement=True,
)

train_transform, val_transform = get_advanced_transforms(150)
train_table.map(TableSampleMapper(train_transform))
val_table.map(TableSampleMapper(val_transform))

train_loader = DataLoader(train_table, batch_size=32, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(val_table, batch_size=32, shuffle=False, num_workers=0)

# 2. Load balanced checkpoint
model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
init_ckpt = PROJECT_ROOT / "scratch/clean_model_seed42_v9_balanced.pth"
model.load_state_dict(torch.load(init_ckpt, map_location=device))
print(f"[OK] Loaded initial checkpoint from {init_ckpt.name}")

# 3. Setup SWA
swa_model = AveragedModel(model).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=2.5e-5, weight_decay=1e-4)
swa_scheduler = SWALR(optimizer, swa_lr=2.5e-5)

swa_epochs = 6
print(f"Running SWA for {swa_epochs} epochs on balanced train_0009 (lr=2.5e-5)...")

for epoch in range(swa_epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    swa_model.update_parameters(model)
    swa_scheduler.step()
    print(f"  SWA Epoch {epoch+1}/{swa_epochs} complete. Avg Loss: {total_loss/len(train_loader):.4f}")

# 4. Update BatchNorm
print("\nUpdating SWA BatchNorm statistics...")
update_bn(train_loader, swa_model, device=device)

out_ckpt = PROJECT_ROOT / "scratch/clean_model_seed42_v9_balanced_swa.pth"
torch.save(swa_model.module.state_dict(), out_ckpt)
print(f"[OK] Saved to {out_ckpt.name}")


# 5. Evaluate on Validation with 2-Scale TTA
print("\nEvaluating on Validation Set with 2-Scale TTA...")
swa_model.eval()
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
tf_zoom = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])

# Raw val images for clean 2-scale eval
val_raw = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()
raw_samples = [(val_table[i]["image"], val_table[i]["label"]) for i in range(len(val_table))]
targets = np.array([s[1] for s in raw_samples])

from PIL import Image
p_std_list, p_zoom_list = [], []
with torch.no_grad():
    for img_path, _ in raw_samples:
        img = Image.open(img_path).convert("RGB")
        out1 = swa_model(tf_std(img).unsqueeze(0).to(device))
        out2 = swa_model(tf_zoom(img).unsqueeze(0).to(device))
        p_std_list.append(F.softmax(out1, dim=1).cpu().numpy())
        p_zoom_list.append(F.softmax(out2, dim=1).cpu().numpy())

p_std = np.concatenate(p_std_list, axis=0)
p_zoom = np.concatenate(p_zoom_list, axis=0)
p_2scale = 0.5 * (p_std + p_zoom)

acc_std = 100.0 * (p_std.argmax(1) == targets).mean()
acc_2scale = 100.0 * (p_2scale.argmax(1) == targets).mean()
rec = recall_score(targets, p_2scale.argmax(1), average=None) * 100

print("=" * 70)
print(f"BALANCED SWA MODEL ACCURACY:")
print(f"  Standard 150x150: {acc_std:.2f}%")
print(f"  2-Scale TTA     : {acc_2scale:.2f}% (Pre-SWA was 79.83%)")
print("=" * 70)
for c, r in zip(CLASSES, rec):
    print(f"  {c:10s}: Recall={r:.1f}%")
print("=" * 70)

out_ckpt = PROJECT_ROOT / "scratch/clean_model_seed42_v9_balanced_swa.pth"
torch.save(swa_model.module.state_dict(), out_ckpt)
print(f"[OK] Saved to {out_ckpt.name}")
