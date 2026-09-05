import sys
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

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
tf_zoom = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])

test_dataset_std = TestDataset(test_dir, transform=tf_std, image_size=150)
test_dataset_zoom = TestDataset(test_dir, transform=tf_zoom, image_size=150)

loader_std = DataLoader(test_dataset_std, batch_size=32, shuffle=False, num_workers=2)
loader_zoom = DataLoader(test_dataset_zoom, batch_size=32, shuffle=False, num_workers=2)

def get_test_logits(ckpt_path):
    print(f"  Inferring 2-Scale TTA on test set: {ckpt_path}")
    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    logits_std, logits_zoom = [], []
    with torch.no_grad():
        for imgs, _ in loader_std:
            logits_std.append(model(imgs.to(device)).cpu().numpy())
        for imgs, _ in loader_zoom:
            logits_zoom.append(model(imgs.to(device)).cpu().numpy())
    l_s = np.concatenate(logits_std, axis=0)
    l_z = np.concatenate(logits_zoom, axis=0)
    return 0.5 * (l_s + l_z)

print("Starting 6-Way Grand Stack Test-Set Inference with 2-Scale Multi-Crop TTA...")

# 1. v5 ensemble
l_v5_42 = get_test_logits("scratch/clean_model_seed42_v5.pth")
l_v5_43 = get_test_logits("scratch/clean_model_seed43_v5.pth")
l_v5_44 = get_test_logits("scratch/clean_model_seed44_v5.pth")
l_v5 = (l_v5_42 + l_v5_43 + l_v5_44) / 3.0

# 2. v9 standard ensemble
l_v9_42 = get_test_logits("scratch/clean_model_seed42_v9.pth")
l_v9_43 = get_test_logits("scratch/clean_model_seed43_v9.pth")
l_v9_44 = get_test_logits("scratch/clean_model_seed44_v9.pth")
l_v9 = (l_v9_42 + l_v9_43 + l_v9_44) / 3.0

# 3. v9 balanced 1x
l_bal1 = get_test_logits("scratch/clean_model_seed42_v9_balanced.pth")

# 4. v9 balanced 2x
l_bal2 = get_test_logits("scratch/clean_model_seed43_v9_glacier2x.pth")

# 5. v9 trivial
l_triv = get_test_logits("scratch/clean_model_seed42_v9_trivial.pth")

# 6. v9 SWA
l_swa = get_test_logits("scratch/clean_model_seed42_v9_swa.pth")

# Apply mean temperatures: [0.94, 0.91, 0.94, 0.89, 0.89, 0.90]
temps = [0.94, 0.91, 0.94, 0.89, 0.89, 0.90]
models_logits = [l_v5, l_v9, l_bal1, l_triv, l_swa, l_bal2]

calib_probs = [F.softmax(torch.tensor(models_logits[i] / temps[i]), dim=1).numpy() for i in range(6)]

# Mean OOF optimal weights from 5-fold CV:
# [0.15 v5, 0.12 v9, 0.18 bal1, 0.18 triv, 0.22 swa, 0.15 bal2]
weights = [0.15, 0.12, 0.18, 0.18, 0.22, 0.15]
w_sum = sum(weights)
weights = [w / w_sum for w in weights]

p_grand = sum(weights[i] * calib_probs[i] for i in range(6))

image_ids = [p.stem for p in test_dataset_std.images]
sample_df = pd.read_csv(sample_sub_path)

out_df = pd.DataFrame({
    "image_id": image_ids,
    "prediction": p_grand.argmax(axis=1),
    "confidence": p_grand.max(axis=1)
}).set_index("image_id").reindex(sample_df["image_id"]).reset_index()

# Save final submissions
out_path = PROJECT_ROOT / "submission.csv"
backup_path = PROJECT_ROOT / "submission_grand_stack_8125.csv"
timestamp_path = PROJECT_ROOT / f"submissions/submission_grand_stack_8125_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

out_df.to_csv(out_path, index=False)
out_df.to_csv(backup_path, index=False)
out_df.to_csv(timestamp_path, index=False)

print("\n" + "=" * 70)
print("[OK] Grand Stack Submission Saved Successfully:")
print(f"  - {out_path}")
print(f"  - {backup_path}")
print(f"  - {timestamp_path}")
print("=" * 70)

print("Prediction distribution:")
for idx, c in enumerate(CLASSES):
    count = (out_df["prediction"] == idx).sum()
    pct = 100.0 * count / len(out_df)
    print(f"  {c:10s} (class {idx}): {count:4d} ({pct:.1f}%)")
