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
from sklearn.metrics import recall_score
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, load_config
from src.model import ResNet18Classifier, verify_from_scratch
from src.dataset import TableSampleMapper
from src.augment import get_advanced_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

set_seed(42)
config = load_config()

# 1. Focal Loss Definition
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# 2. Load Tables
print("[1/5] Loading train_0009 and val tables...")
train_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="train_0009").latest()
val_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()

train_transform, val_transform = get_advanced_transforms(150)
train_table.map(TableSampleMapper(train_transform))
val_table.map(TableSampleMapper(val_transform))

train_sampler = train_table.create_sampler(exclude_zero_weights=True)
train_loader = DataLoader(train_table, batch_size=32, sampler=train_sampler, num_workers=2)
val_loader = DataLoader(val_table, batch_size=32, shuffle=False, num_workers=2)

# 3. Model from scratch
print("[2/5] Initializing ResNet-18 strictly from scratch...")
model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
verify_from_scratch(model)

# 4. Optimizer, Scheduler, Focal Loss
epochs = 15
lr = 0.0001
class_weights = torch.tensor([1.0, 1.0, 1.5, 1.3, 1.0, 1.0]).to(device)
criterion = FocalLoss(gamma=2.0, weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# Cosine warmup
warmup_epochs = 3
min_lr = 1e-6
def lr_lambda(epoch_idx):
    if epoch_idx < warmup_epochs:
        return float(epoch_idx + 1) / float(max(1, warmup_epochs))
    progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
    return max(min_lr / lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

print(f"[3/5] Training for {epochs} epochs with Focal Loss (gamma=2.0, glacier=1.5, mountain=1.3)...")
best_val_acc = 0.0
save_ckpt = PROJECT_ROOT / "scratch/clean_model_seed42_v9_focal.pth"

for epoch in range(epochs):
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
    
    scheduler.step()
    
    # Quick val check
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = 100.0 * val_correct / val_total
    print(f"  Epoch {epoch+1:2d}/{epochs:2d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% (LR: {scheduler.get_last_lr()[0]:.6f})")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_ckpt)

print(f"\n[4/5] Best Val Accuracy during training: {best_val_acc:.2f}%")

# 5. Evaluate with 2-Scale Multi-Crop TTA
print("\n[5/5] Evaluating Best Focal Model with 2-Scale Multi-Crop TTA...")
model.load_state_dict(torch.load(save_ckpt, map_location=device))
model.eval()

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
tf_zoom = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])

t_val_std = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()
t_val_zoom = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()
t_val_std.map(TableSampleMapper(tf_std))
t_val_zoom.map(TableSampleMapper(tf_zoom))

loader_std = DataLoader(t_val_std, batch_size=32, shuffle=False, num_workers=2)
loader_zoom = DataLoader(t_val_zoom, batch_size=32, shuffle=False, num_workers=2)

p_std_list, p_zoom_list, targets = [], [], []
with torch.no_grad():
    for imgs, lbls in loader_std:
        p_std_list.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
        targets.extend(lbls.numpy())
    for imgs, _ in loader_zoom:
        p_zoom_list.append(F.softmax(model(imgs.to(device)), dim=1).cpu().numpy())

p_std = np.concatenate(p_std_list, axis=0)
p_zoom = np.concatenate(p_zoom_list, axis=0)
p_2scale = 0.5 * (p_std + p_zoom)
targets = np.array(targets)

acc_std = 100.0 * (p_std.argmax(1) == targets).mean()
acc_2scale = 100.0 * (p_2scale.argmax(1) == targets).mean()
rec = recall_score(targets, p_2scale.argmax(1), average=None) * 100

print("=" * 70)
print(f"FOCAL LOSS MODEL EVALUATION:")
print(f"  Standard 150x150 : {acc_std:.2f}%")
print(f"  2-Scale TTA      : {acc_2scale:.2f}%")
print("=" * 70)
for c, r in zip(CLASSES, rec):
    print(f"  {c:10s}: Recall={r:.1f}%")
print("=" * 70)
print(f"[OK] Saved to {save_ckpt.name}")
