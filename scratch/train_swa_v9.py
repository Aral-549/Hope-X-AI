import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import recall_score
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, load_config
from src.model import ResNet18Classifier
from src.dataset import TableSampleMapper
from src.augment import get_trivialaugment_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

set_seed(42)
config = load_config()

# 1. Load Tables
train_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="train_0009").latest()
val_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()

train_transform, val_transform = get_trivialaugment_transforms(150)

train_table.map(TableSampleMapper(train_transform))
val_table.map(TableSampleMapper(val_transform))

train_sampler = train_table.create_sampler(exclude_zero_weights=True)
train_loader = DataLoader(train_table, batch_size=32, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(val_table, batch_size=32, shuffle=False, num_workers=0)

# 2. Load the best single model (TrivialAugment Seed 42 - 79.42%)
model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
init_ckpt = PROJECT_ROOT / "scratch/clean_model_seed42_v9_trivial.pth"
model.load_state_dict(torch.load(init_ckpt, map_location=device))
print(f"[OK] Loaded initial checkpoint from {init_ckpt.name}")

# 3. Setup SWA
swa_model = AveragedModel(model).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
swa_scheduler = SWALR(optimizer, swa_lr=3e-5)

swa_epochs = 6
print(f"Running SWA for {swa_epochs} epochs on train_0009 (lr=3e-5)...")

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

# 4. Update BatchNorm statistics using training data
print("\nUpdating SWA BatchNorm statistics on train set...")
update_bn(train_loader, swa_model, device=device)

# 5. Evaluate SWA Model on Validation Set
print("\nEvaluating SWA Model on Validation Set...")
swa_model.eval()
val_correct = 0
val_total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = swa_model(images)
        preds = outputs.argmax(1).cpu()
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)
        all_preds.extend(preds.numpy())
        all_targets.extend(labels.numpy())

swa_val_acc = 100.0 * val_correct / val_total
rec_swa = recall_score(all_targets, all_preds, average=None) * 100

print("=" * 70)
print(f"  SWA MODEL VALIDATION ACCURACY: {swa_val_acc:.2f}% (vs 79.42% pre-SWA)")
print("=" * 70)
for c, r in zip(CLASSES, rec_swa):
    print(f"  {c:10s}: Recall={r:.1f}%")
print("=" * 70)

# 6. Save checkpoint
swa_ckpt = PROJECT_ROOT / "scratch/clean_model_seed42_v9_swa.pth"
torch.save(swa_model.module.state_dict(), swa_ckpt)
print(f"[OK] SWA Checkpoint saved to {swa_ckpt}")
