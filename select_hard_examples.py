"""
select_hard_examples.py
=======================
HackBlox 2026 · 3LC Scene Classification Challenge
Selects the hardest samples from the remaining unlabeled pool of train_0005
uniformly across all 6 categories using ensemble prediction entropy.

Remaining labeling budget: 3000 - 2498 = 502 samples (~83-84 per class).
"""

import sys
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ResNet18Classifier

# ── CONFIG ────────────────────────────────────────────────────────────
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

CHECKPOINT_PATHS = [
    PROJECT_ROOT / "scratch/clean_model_seed42_v5.pth",
    PROJECT_ROOT / "scratch/clean_model_seed43_v5.pth",
    PROJECT_ROOT / "scratch/clean_model_seed44_v5.pth",
]

BASE_TABLE_URL = "/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0005"
OUTPUT_CSV = PROJECT_ROOT / "scratch/hard_examples_for_review.csv"
OUTPUT_JSON = PROJECT_ROOT / "scratch/hard_examples_queue.json"
TOTAL_BUDGET = 502          # 3000 max budget - 2498 active in train_0005
IMG_SIZE = 150              # Competition resolution
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────


class UnlabeledTableDataset(Dataset):
    """Pulls unlabeled samples directly from 3LC table train_0005."""
    def __init__(self, table_url: str, img_size: int):
        table = tlc.Table.from_url(table_url)
        self.rows = []
        for i, row in enumerate(table.table_rows):
            if row.get("weight", 0.0) == 0.0:
                self.rows.append((row["id"], table[i]["image"]))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print(f"[DATA] Loaded {len(self.rows)} unlabeled candidate images from {Path(table_url).name}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sample_id, path = self.rows[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Error reading {path}: {e}")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return sample_id, self.transform(img), path


def load_model(checkpoint_path: Path, num_classes: int = 6):
    """Load scratch ResNet-18 model checkpoint."""
    model = ResNet18Classifier(num_classes=num_classes, dropout_rate=0.3)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    print("=" * 70)
    print("  Selecting Hardest Examples Uniformly Across Classes")
    print(f"  Target Budget: {TOTAL_BUDGET} samples (~{TOTAL_BUDGET // len(CLASS_NAMES)} per class)")
    print("=" * 70)

    dataset = UnlabeledTableDataset(BASE_TABLE_URL, IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    models = [load_model(p) for p in CHECKPOINT_PATHS]
    print(f"Loaded {len(models)} ensemble member(s) on {DEVICE}")

    all_ids, all_paths, all_probs = [], [], []
    with torch.no_grad():
        for sample_ids, imgs, paths in loader:
            imgs = imgs.to(DEVICE)
            batch_probs = torch.zeros(imgs.size(0), len(CLASS_NAMES), device=DEVICE)
            for model in models:
                batch_probs += F.softmax(model(imgs), dim=1)
            batch_probs /= len(models)

            all_ids.extend([int(x) for x in sample_ids])
            all_paths.extend(paths)
            all_probs.append(batch_probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)  # [N, num_classes]

    # Uncertainty = entropy of ensemble averaged softmax
    # High entropy = maximum confusion / disagreement among models
    entropy = -(all_probs * torch.log(all_probs.clamp_min(1e-9))).sum(dim=1)
    pred_class = all_probs.argmax(dim=1)
    conf = all_probs.max(dim=1).values

    # Top-2 class indices for reference
    sorted_indices = torch.argsort(all_probs, dim=1, descending=True)
    top2_class = sorted_indices[:, 1]
    margin = (all_probs.gather(1, sorted_indices[:, 0:1]) - all_probs.gather(1, sorted_indices[:, 1:2])).squeeze(1)

    # Bucket by model-PREDICTED class, hardest first
    buckets = defaultdict(list)
    for i, (sample_id, path, cls_idx, ent, c, m, t2) in enumerate(
        zip(all_ids, all_paths, pred_class.tolist(), entropy.tolist(), conf.tolist(), margin.tolist(), top2_class.tolist())
    ):
        buckets[cls_idx].append({
            "id": sample_id,
            "image_path": path,
            "filename": Path(path).name,
            "predicted_class_idx": cls_idx,
            "predicted_class": CLASS_NAMES[cls_idx],
            "top2_class_idx": t2,
            "top2_class": CLASS_NAMES[t2],
            "entropy": float(ent),
            "confidence": float(c),
            "margin": float(m),
        })

    # Sort each bucket by entropy descending (hardest / most confused first)
    for cls_idx in buckets:
        buckets[cls_idx].sort(key=lambda x: x["entropy"], reverse=True)

    per_class_budget = TOTAL_BUDGET // len(CLASS_NAMES)  # 83
    remainder = TOTAL_BUDGET - per_class_budget * len(CLASS_NAMES)  # 4

    selected = []
    # Prioritize remainder for weaker / under-represented classes (buildings, sea, glacier, mountain)
    priority_classes = [0, 2, 3, 4, 5, 1]
    remainder_allocated = 0

    for cls_idx in range(len(CLASS_NAMES)):
        extra = 1 if remainder_allocated < remainder and cls_idx in priority_classes[:remainder] else 0
        if extra:
            remainder_allocated += 1
        take = per_class_budget + extra

        picks = buckets.get(cls_idx, [])[:take]
        if len(picks) < take:
            print(f"WARNING: only {len(picks)} candidates for '{CLASS_NAMES[cls_idx]}', wanted {take}")

        for item in picks:
            item["quota_class"] = CLASS_NAMES[cls_idx]
            selected.append(item)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["id", "image_path", "filename", "model_predicted_class", "top2_class", "entropy", "confidence", "margin"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "id": row["id"],
                "image_path": row["image_path"],
                "filename": row["filename"],
                "model_predicted_class": row["predicted_class"],
                "top2_class": row["top2_class"],
                "entropy": round(row["entropy"], 4),
                "confidence": round(row["confidence"], 4),
                "margin": round(row["margin"], 4),
            })

    # Write JSON queue for CLI review
    with open(OUTPUT_JSON, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\n[OK] Wrote {len(selected)} hard examples to {OUTPUT_CSV}")
    print(f"[OK] Wrote JSON review queue to {OUTPUT_JSON}")

    print("\nPer predicted-class counts (balanced across categories):")
    counts = defaultdict(int)
    for row in selected:
        counts[row["predicted_class"]] += 1
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Class {i} ({name:10s}): {counts[name]:2d} samples")

    print("\n" + "=" * 70)
    print(f"  [SUCCESS] Selected exactly {len(selected)} hard boundary cases.")
    print("=" * 70)


if __name__ == "__main__":
    main()
