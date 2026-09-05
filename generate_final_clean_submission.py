import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, load_config
from src.model import ResNet18Classifier
from src.dataset import TestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

config = load_config()
set_seed(42)

test_dir = PROJECT_ROOT / config["data"]["paths"]["test_dir"]
sample_sub_path = PROJECT_ROOT / config["data"]["paths"]["sample_submission"]

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

test_dataset_base = TestDataset(test_dir, transform=tf_base, image_size=150)
test_dataset_flip = TestDataset(test_dir, transform=tf_flip, image_size=150)

loader_base = DataLoader(test_dataset_base, batch_size=32, shuffle=False, num_workers=2)
loader_flip = DataLoader(test_dataset_flip, batch_size=32, shuffle=False, num_workers=2)

def get_test_probs(ckpt_path):
    print(f"  Inferring on test set with: {ckpt_path}")
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    probs_base = []
    with torch.no_grad():
        for imgs, _ in loader_base:
            imgs = imgs.to(device)
            p = F.softmax(model(imgs), dim=1).cpu().numpy()
            probs_base.append(p)
    p_base = np.concatenate(probs_base, axis=0)

    probs_flip = []
    with torch.no_grad():
        for imgs, _ in loader_flip:
            imgs = imgs.to(device)
            p = F.softmax(model(imgs), dim=1).cpu().numpy()
            probs_flip.append(p)
    p_flip = np.concatenate(probs_flip, axis=0)

    return 0.5 * (p_base + p_flip)

print("Starting ensemble inference for 80.92% clean stack...")

# 1. v5 ensemble
ckpts_v5 = [
    "scratch/clean_model_seed42_v5.pth",
    "scratch/clean_model_seed43_v5.pth",
    "scratch/clean_model_seed44_v5.pth",
]
p_v5 = np.mean([get_test_probs(c) for c in ckpts_v5], axis=0)

# 2. v9 standard ensemble
ckpts_v9 = [
    "scratch/clean_model_seed42_v9.pth",
    "scratch/clean_model_seed43_v9.pth",
    "scratch/clean_model_seed44_v9.pth",
]
p_v9 = np.mean([get_test_probs(c) for c in ckpts_v9], axis=0)

# 3. v9 balanced
p_bal = get_test_probs("scratch/clean_model_seed42_v9_balanced.pth")

# 4. v9 trivial augment
p_triv = get_test_probs("scratch/clean_model_seed42_v9_trivial.pth")

# Blend with optimal weights: [0.368, 0.263, 0.105, 0.263]
p_blend = 0.368 * p_v5 + 0.263 * p_v9 + 0.105 * p_bal + 0.263 * p_triv

confidences = p_blend.max(axis=1)
predictions = p_blend.argmax(axis=1)

# Get image IDs in order (stem is image_id e.g. "test_00001")
image_ids = [p.stem for p in test_dataset_base.images]

out_df = pd.DataFrame({
    "image_id": image_ids,
    "prediction": predictions,
    "confidence": confidences
})

# Verify alignment with sample_submission.csv
sample_df = pd.read_csv(sample_sub_path)
out_df = out_df.set_index("image_id").reindex(sample_df["image_id"]).reset_index()

out_path = PROJECT_ROOT / "submission.csv"
backup_path = PROJECT_ROOT / "submission_clean_8092.csv"
timestamp_path = PROJECT_ROOT / f"submissions/submission_clean_8092_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

out_df.to_csv(out_path, index=False)
out_df.to_csv(backup_path, index=False)
out_df.to_csv(timestamp_path, index=False)

print(f"\n[OK] Submission saved successfully to:")
print(f"  - {out_path}")
print(f"  - {backup_path}")
print(f"  - {timestamp_path}")

print("\nPrediction distribution:")
for idx, c in enumerate(CLASSES):
    count = (out_df["prediction"] == idx).sum()
    pct = 100.0 * count / len(out_df)
    print(f"  {c:10s} (class {idx}): {count:4d} ({pct:.1f}%)")
