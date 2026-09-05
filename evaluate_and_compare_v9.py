"""
evaluate_and_compare_v9.py
==========================
Evaluates the authentic human-reviewed 3-seed ensemble (train_0009)
against the train_0005 baseline.
"""

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

def evaluate_models(ckpts, loader):
    models = []
    for p in ckpts:
        p_path = PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)
        if not p_path.exists():
            print(f"[ERROR] Checkpoint not found: {p_path}")
            return None, None
        m = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
        m.load_state_dict(torch.load(p_path, map_location=device))
        m.eval()
        models.append(m)

    all_probs = [[] for _ in models]
    targets = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            targets.extend(lbls.numpy())
            for m_idx, m in enumerate(models):
                all_probs[m_idx].append(F.softmax(m(imgs), dim=1).cpu().numpy())

    probs_cat = [np.concatenate(p, axis=0) for p in all_probs]
    ens_probs = np.mean(probs_cat, axis=0)
    return ens_probs, np.array(targets)

def main():
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
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.transform(Image.open(self.samples[idx][0]).convert("RGB")), self.samples[idx][1]

    val_loader = DataLoader(ValDataset(val_table, tf), batch_size=32, shuffle=False, num_workers=2)

    ckpts_v5 = [
        "scratch/clean_model_seed42_v5.pth",
        "scratch/clean_model_seed43_v5.pth",
        "scratch/clean_model_seed44_v5.pth",
    ]
    ckpts_v9 = [
        "scratch/clean_model_seed42_v9.pth",
        "scratch/clean_model_seed43_v9.pth",
        "scratch/clean_model_seed44_v9.pth",
    ]

    print("=" * 70)
    print("  Evaluating Authentic Human Ensemble (train_0009) vs Baseline (train_0005)")
    print("=" * 70)

    p_v5, targets = evaluate_models(ckpts_v5, val_loader)
    p_v9, _ = evaluate_models(ckpts_v9, val_loader)

    if p_v9 is None:
        print("[ERROR] train_0009 models not ready yet.")
        sys.exit(1)

    acc_v5 = 100.0 * (p_v5.argmax(axis=1) == targets).mean()
    acc_v9 = 100.0 * (p_v9.argmax(axis=1) == targets).mean()

    rec_v5 = recall_score(targets, p_v5.argmax(axis=1), average=None) * 100
    rec_v9 = recall_score(targets, p_v9.argmax(axis=1), average=None) * 100

    delta_v9_v5 = rec_v9 - rec_v5

    df_comp = pd.DataFrame({
        "Class": CLASSES,
        "train_0005 (Base)": rec_v5.round(1),
        "train_0009 (Human 3000)": rec_v9.round(1),
        "Delta": delta_v9_v5.round(1),
    })

    print("\n" + "=" * 70)
    print("  PER-CLASS RECALL (%) COMPARISON")
    print("=" * 70)
    print(df_comp.to_string(index=False))
    print("-" * 70)
    print(f"Overall Accuracy:  train_0005: {acc_v5:.2f}%  -->  train_0009: {acc_v9:.2f}%")
    print("=" * 70)

    df_comp.to_csv(PROJECT_ROOT / "scratch/per_class_recall_comparison_v9.csv", index=False)

    print("\nDetailed Classification Report for Authentic train_0009 3-seed Ensemble:")
    print(classification_report(targets, p_v9.argmax(axis=1), target_names=CLASSES, digits=4))

if __name__ == "__main__":
    main()
