"""
evaluate_and_compare_v6b.py
===========================
Evaluates 3-seed ensembles for:
- train_0005 (baseline clean)
- train_0006 (noisy buildings/street pseudo-labels)
- train_0006b (surgical fix: mountain/glacier/sea/forest only)

Outputs the complete per-class recall comparison table and generates submission.
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
from src.dataset import TestDataset
from src.augment import get_test_transforms
from src.utils import load_config

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
    config = load_config()
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
    ckpts_v6 = [
        "scratch/clean_model_seed42_v6.pth",
        "scratch/clean_model_seed43_v6.pth",
        "scratch/clean_model_seed44_v6.pth",
    ]
    ckpts_v6b = [
        "scratch/clean_model_seed42_v6b.pth",
        "scratch/clean_model_seed43_v6b.pth",
        "scratch/clean_model_seed44_v6b.pth",
    ]

    print("=" * 70)
    print("  Evaluating 3-Seed Ensembles: train_0005 vs train_0006 vs train_0006b")
    print("=" * 70)

    p_v5, targets = evaluate_models(ckpts_v5, val_loader)
    p_v6, _ = evaluate_models(ckpts_v6, val_loader)
    p_v6b, _ = evaluate_models(ckpts_v6b, val_loader)

    if p_v6b is None:
        print("[ERROR] train_0006b models not ready yet.")
        sys.exit(1)

    acc_v5 = 100.0 * (p_v5.argmax(axis=1) == targets).mean()
    acc_v6 = 100.0 * (p_v6.argmax(axis=1) == targets).mean()
    acc_v6b = 100.0 * (p_v6b.argmax(axis=1) == targets).mean()

    rec_v5 = recall_score(targets, p_v5.argmax(axis=1), average=None) * 100
    rec_v6 = recall_score(targets, p_v6.argmax(axis=1), average=None) * 100
    rec_v6b = recall_score(targets, p_v6b.argmax(axis=1), average=None) * 100

    delta_v6_v5 = rec_v6 - rec_v5
    delta_v6b_v5 = rec_v6b - rec_v5
    delta_v6b_v6 = rec_v6b - rec_v6

    df_comp = pd.DataFrame({
        "Class": CLASSES,
        "train_0005": rec_v5.round(1),
        "train_0006": rec_v6.round(1),
        "train_0006b": rec_v6b.round(1),
        "Delta (0006b - 0005)": delta_v6b_v5.round(1),
        "Delta (0006b - 0006)": delta_v6b_v6.round(1),
    })

    print("\n" + "=" * 70)
    print("  PER-CLASS RECALL (%) COMPARISON")
    print("=" * 70)
    print(df_comp.to_string(index=False))
    print("-" * 70)
    print(f"Overall Accuracy:  train_0005: {acc_v5:.2f}%  |  train_0006: {acc_v6:.2f}%  |  train_0006b: {acc_v6b:.2f}%")
    print("=" * 70)

    # Save summary dataframe
    df_comp.to_csv(PROJECT_ROOT / "scratch/per_class_recall_comparison.csv", index=False)

    # Detailed classification report for 6b
    print("\nDetailed Classification Report for train_0006b 3-seed Ensemble:")
    print(classification_report(targets, p_v6b.argmax(axis=1), target_names=CLASSES, digits=4))

if __name__ == "__main__":
    main()
