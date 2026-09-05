"""
Generate test set predictions and format submission.csv for Kaggle.
Reproducible, verified against sample_submission.csv format rules.
"""

import sys
import csv
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, load_config
from src.model import ResNet18Classifier
from src.dataset import TestDataset
from src.augment import get_test_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    config = load_config()
    set_seed(config["project"]["random_seed"])

    print("=" * 70)
    print("  Generating Kaggle Submission Predictions")
    print("=" * 70)

    model_path = PROJECT_ROOT / config["data"]["paths"]["best_model_path"]
    test_dir = PROJECT_ROOT / config["data"]["paths"]["test_dir"]
    sample_sub_path = PROJECT_ROOT / config["data"]["paths"]["sample_submission"]
    output_path = PROJECT_ROOT / config["data"]["paths"]["submission_output"]
    subs_dir = PROJECT_ROOT / config["data"]["paths"]["submissions_dir"]

    if not model_path.exists():
        print(f"[FATAL] Model checkpoint not found at {model_path}")
        sys.exit(1)

    if not test_dir.exists():
        print(f"[FATAL] Test directory not found at {test_dir}")
        sys.exit(1)

    # 1. Load model
    print(f"\n[1/4] Loading model checkpoint from {model_path}...")
    model = ResNet18Classifier(
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"],
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  [OK] Model successfully loaded.")

    # 2. Test dataset and DataLoader
    print(f"\n[2/4] Preparing test dataset from {test_dir}...")
    test_transform = get_test_transforms(config["data"]["image_size"])
    test_dataset = TestDataset(test_dir, transform=test_transform, image_size=config["data"]["image_size"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # 3. Predict
    print(f"\n[3/4] Running inference on {len(test_dataset)} test samples...")
    predictions = {}
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = probs.max(1)
            for img_id, pred, conf in zip(image_ids, preds.cpu().numpy(), confidences.cpu().numpy()):
                predictions[img_id] = {
                    "image_id": img_id,
                    "prediction": int(pred),
                    "confidence": float(conf),
                }

    # 4. Align with sample_submission.csv
    print(f"\n[4/4] Validating and formatting submission against {sample_sub_path}...")
    expected_ids = []
    if sample_sub_path.exists():
        with open(sample_sub_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_ids = [row["image_id"] for row in reader]

    if expected_ids:
        print(f"  Verifying {len(expected_ids)} expected IDs...")
        aligned_rows = []
        for img_id in expected_ids:
            if img_id in predictions:
                aligned_rows.append(predictions[img_id])
            else:
                print(f"  [WARN] Missing prediction for {img_id}, applying fallback.")
                aligned_rows.append({"image_id": img_id, "prediction": 0, "confidence": 0.5})
    else:
        aligned_rows = list(predictions.values())

    # Write output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(aligned_rows)

    print(f"  [OK] Saved submission: {output_path} ({len(aligned_rows)} rows)")

    # Save timestamped backup
    subs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = subs_dir / f"submission_{timestamp}.csv"
    shutil.copyfile(output_path, backup_path)
    print(f"  [OK] Saved timestamped backup: {backup_path}")

    # Validate output
    assert len(aligned_rows) == 1800, f"Expected 1800 rows, got {len(aligned_rows)}"
    for r in aligned_rows:
        assert 0 <= r["prediction"] <= 5, f"Invalid class label: {r['prediction']}"
        assert 0.0 <= r["confidence"] <= 1.0, f"Confidence out of range: {r['confidence']}"
    print("  [VALIDATION PASSED] All 1,800 test predictions meet competition rules.")
    print("=" * 70)


if __name__ == "__main__":
    main()
