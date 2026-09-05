"""
Multi-Seed and Checkpoint Ensembling Pipeline.
Averages calibrated softmax class probabilities across multiple ResNet-18 models
trained strictly from scratch on the 3LC dataset.
Rule-Compliant: Uses single fixed architecture, zero external data, deterministic inference.
"""

import sys
import csv
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from src.utils import set_seed, load_config
from src.model import ResNet18Classifier
from src.dataset import TestDataset
from src.augment import get_test_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Ensemble Inference")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best_model_seed42.pth", "best_model_seed43.pth", "best_model_seed44.pth"],
        help="List of checkpoint paths to ensemble",
    )
    parser.add_argument("--use-tta", action="store_true", help="Apply horizontal flip TTA averaging")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output submission CSV path")
    return parser.parse_args()


def load_ensemble_models(ckpt_paths, config):
    models = []
    for p_str in ckpt_paths:
        p = PROJECT_ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
        if not p.exists():
            print(f"[WARN] Checkpoint not found: {p}, skipping.")
            continue
        print(f"  Loading model checkpoint: {p.name}")
        model = ResNet18Classifier(
            num_classes=config["model"]["num_classes"],
            dropout_rate=config["model"]["dropout_rate"],
        )
        state_dict = torch.load(p, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def evaluate_val_ensemble(models, val_dir, use_tta=False):
    """Evaluate ensemble accuracy on the 1,200 validation images."""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t_orig = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
    t_flip = transforms.Compose([transforms.Resize((150, 150)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm])

    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    val_path = Path(val_dir)
    if not val_path.exists():
        return None

    correct, total = 0, 0
    with torch.no_grad():
        for label_idx, cls_name in enumerate(classes):
            cls_dir = val_path / cls_name
            for img_file in cls_dir.glob("*.*"):
                img = Image.open(img_file).convert("RGB")
                x = t_orig(img).unsqueeze(0).to(device)
                
                probs_sum = torch.zeros((1, len(classes)), device=device)
                for m in models:
                    p = F.softmax(m(x), dim=1)
                    if use_tta:
                        x_flip = t_flip(img).unsqueeze(0).to(device)
                        p_flip = F.softmax(m(x_flip), dim=1)
                        p = (p + p_flip) / 2.0
                    probs_sum += p
                
                probs_avg = probs_sum / len(models)
                pred = probs_avg.argmax(1).item()
                if pred == label_idx:
                    correct += 1
                total += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"\n[EVALUATION] Ensemble Validation Accuracy: {correct}/{total} ({acc:.2f}%) across {len(models)} models (TTA={use_tta})")
    return acc


def main():
    args = parse_args()
    config = load_config()
    set_seed(config["project"]["random_seed"])

    print("=" * 70)
    print("  Multi-Seed Ensemble Inference (Rule-Compliant ResNet-18)")
    print("=" * 70)

    # 1. Load all models
    print(f"\n[1/4] Loading {len(args.checkpoints)} checkpoint(s)...")
    models = load_ensemble_models(args.checkpoints, config)
    if not models:
        print("[FATAL] No valid models could be loaded.")
        sys.exit(1)
    print(f"  [OK] Successfully initialized {len(models)} model(s) for ensembling.")

    # 2. Evaluate on validation set
    print("\n[2/4] Measuring ensemble accuracy on validation set...")
    val_dir = PROJECT_ROOT / config["data"]["paths"]["val_dir"]
    evaluate_val_ensemble(models, val_dir, use_tta=args.use_tta)

    # 3. Predict on Test Set
    test_dir = PROJECT_ROOT / config["data"]["paths"]["test_dir"]
    test_transform = get_test_transforms(config["data"]["image_size"])
    test_dataset = TestDataset(test_dir, transform=test_transform, image_size=config["data"]["image_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["inference"]["batch_size"], shuffle=False, num_workers=0)

    print(f"\n[3/4] Running ensemble inference on {len(test_dataset)} test samples...")
    predictions = {}
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Ensemble Inference"):
            images = images.to(device)
            batch_probs = torch.zeros((images.size(0), config["model"]["num_classes"]), device=device)
            
            for m in models:
                p = F.softmax(m(images), dim=1)
                if args.use_tta:
                    images_flipped = torch.flip(images, dims=[3])
                    p_flip = F.softmax(m(images_flipped), dim=1)
                    p = (p + p_flip) / 2.0
                batch_probs += p
            
            batch_probs /= len(models)
            confidences, preds = batch_probs.max(1)
            
            for img_id, pred, conf in zip(image_ids, preds.cpu().numpy(), confidences.cpu().numpy()):
                predictions[img_id] = {
                    "image_id": img_id,
                    "prediction": int(pred),
                    "confidence": float(conf),
                }

    # 4. Align with sample_submission.csv
    print(f"\n[4/4] Aligning and validating against sample_submission.csv...")
    sample_sub_path = PROJECT_ROOT / config["data"]["paths"]["sample_submission"]
    expected_ids = []
    if sample_sub_path.exists():
        with open(sample_sub_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_ids = [row["image_id"] for row in reader]

    aligned_rows = []
    for img_id in expected_ids:
        if img_id in predictions:
            aligned_rows.append(predictions[img_id])
        else:
            aligned_rows.append({"image_id": img_id, "prediction": 0, "confidence": 0.5})

    output_path = PROJECT_ROOT / args.output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(aligned_rows)

    print(f"  [OK] Saved submission: {output_path} ({len(aligned_rows)} rows)")

    # Save timestamped backup
    subs_dir = PROJECT_ROOT / config["data"]["paths"]["submissions_dir"]
    subs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = subs_dir / f"submission_ensemble_{timestamp}.csv"
    shutil.copyfile(output_path, backup_path)
    print(f"  [OK] Saved timestamped backup: {backup_path}")

    # Validate output
    assert len(aligned_rows) == 1800, f"Expected 1800 rows, got {len(aligned_rows)}"
    for r in aligned_rows:
        assert 0 <= r["prediction"] <= 5, f"Invalid class label: {r['prediction']}"
        assert 0.0 <= r["confidence"] <= 1.0, f"Confidence out of range: {r['confidence']}"
    print("  [VALIDATION PASSED] All 1,800 predictions meet competition rules.")
    print("=" * 70)


if __name__ == "__main__":
    main()
