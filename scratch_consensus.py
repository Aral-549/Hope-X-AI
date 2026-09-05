"""
scratch_consensus.py
====================
HackBlox 2026 · 3LC Scene Classification Challenge
Step 1: Automated Majority Pseudo-Labeling via Scratch-Trained ResNet-18 Ensemble.

STRICT COMPLIANCE RULES:
1. ResNet-18 only, initialized strictly from scratch (weights=None).
2. Trained ONLY on the clean 600 official seed samples (from root 3LC table 'train').
3. Zero external models or pretrained weights (no CLIP, no ImageNet pretraining).
4. Strict consensus: All 3 seeds must agree on predicted class AND each seed confidence >= threshold (0.90).
5. Active sample budget guardrail: Total active samples <= 3,000.
"""

import sys
import argparse
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import tlc

from src.model import ResNet18Classifier, verify_from_scratch
from src.augment import get_advanced_transforms, get_test_transforms
from src.utils import set_seed, load_config

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InMemoryImageDataset(Dataset):
    """Simple, high-performance PyTorch dataset for memory/disk cached image samples."""
    def __init__(self, samples, transform=None):
        """
        samples: list of tuples (image_path, label, row_id)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, row_id = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            img = Image.new("RGB", (150, 150), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, label, row_id


def train_single_seed(
    seed: int,
    train_samples: list,
    val_samples: list,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    save_path: Path = None,
) -> tuple:
    """
    Train a single ResNet-18 strictly from scratch on the 600 seed samples.
    Returns (trained_model, best_val_acc).
    """
    print(f"\n" + "=" * 65)
    print(f"  Training Scratch ResNet-18 | Seed {seed} | Epochs {epochs} | LR {lr}")
    print(f"=" * 65)

    set_seed(seed)
    train_tf, val_tf = get_advanced_transforms(150)

    train_ds = InMemoryImageDataset(train_samples, transform=train_tf)
    val_ds = InMemoryImageDataset(val_samples, transform=val_tf)

    # Note: For 600 seed images, each class has exactly 100 images, perfectly balanced.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(DEVICE)
    verify_from_scratch(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine scheduler with linear warmup (3 epochs)
    warmup_epochs = 3
    min_lr = 1e-6

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        progress = float(ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return max(min_lr / lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * train_correct / total_train
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation
        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100.0 * val_correct / total_val
        print(f"  Seed {seed} | Epoch {epoch+1:2d}/{epochs:2d} (lr={current_lr:.6f}) | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"  [RESULT] Seed {seed} Best Val Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch}")

    if best_model_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  [CHECKPOINT] Saved model to {save_path}")

    return model, best_val_acc


def run_inference_on_pool(
    model: nn.Module,
    pool_samples: list,
    batch_size: int = 32,
) -> dict:
    """
    Run inference on the unlabeled pool and return dict of row_id -> {probs, pred, conf}.
    """
    model.eval()
    test_tf = get_test_transforms(150)
    pool_ds = InMemoryImageDataset(pool_samples, transform=test_tf)
    pool_loader = DataLoader(
        pool_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    results = {}
    with torch.no_grad():
        for imgs, _, row_ids in tqdm(pool_loader, desc="  Inference on pool", leave=False):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)

            for rid, p_dist, pred, conf in zip(row_ids.numpy(), probs, preds, confs):
                results[int(rid)] = {
                    "probs": p_dist.tolist(),
                    "pred": int(pred),
                    "conf": float(conf),
                }

    return results


def main():
    parser = argparse.ArgumentParser(description="Scratch Consensus Pseudo-Labeling")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Ensemble seeds")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per seed")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--conf-thresh", type=float, default=0.90, help="Minimum confidence threshold for consensus")
    parser.add_argument("--max-consensus", type=int, default=1800, help="Maximum consensus samples to auto-label")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if checkpoints exist")
    args = parser.parse_args()

    print("=" * 70)
    print("  HackBlox 2026 · Scratch Consensus Pseudo-Labeling Engine")
    print(f"  Seeds: {args.seeds} | Conf Threshold: {args.conf_thresh:.2f} | Max Budget: {args.max_consensus}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    config = load_config()
    tlc.register_project_url_alias(
        token="INTEL_SCENE_DATA",
        path=str(PROJECT_ROOT.absolute()),
        project=config["project"]["name"],
    )

    # 1. Load Root Train Table (Clean base table with exactly 600 seed samples)
    root_table_url = "/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train"
    print(f"\n[1/5] Loading root clean train table: {root_table_url}")
    train_table = tlc.Table.from_url(root_table_url)

    val_table = tlc.Table.from_names(
        project_name=config["project"]["name"],
        dataset_name=config["project"]["dataset_name"],
        table_name="val",
    ).latest()

    # Extract 600 clean seed samples and 6,000 pool samples
    train_samples = []
    pool_samples = []

    for i, row in enumerate(train_table.table_rows):
        row_dict = dict(row)
        img_path = train_table[i]["image"]
        row_id = row_dict["id"]
        label = row_dict["label"]
        weight = float(row_dict.get("weight", 0.0))

        if weight > 0:
            train_samples.append((img_path, label, row_id))
        else:
            pool_samples.append((img_path, label, row_id))

    print(f"  Clean seed samples: {len(train_samples)} (weight=1.0)")
    print(f"  Unlabeled pool samples: {len(pool_samples)} (weight=0.0)")
    assert len(train_samples) == 600, f"Expected 600 seed samples, got {len(train_samples)}"
    assert len(pool_samples) == 6000, f"Expected 6000 pool samples, got {len(pool_samples)}"

    # Extract val samples
    val_samples = []
    for i, row in enumerate(val_table.table_rows):
        img_path = val_table[i]["image"]
        val_samples.append((img_path, row["label"], row["id"]))
    print(f"  Validation samples: {len(val_samples)}")

    # 2. Train or Load Models for Each Seed
    print("\n[2/5] Training 3 Scratch ResNet-18 models...")
    models_dir = PROJECT_ROOT / "scratch" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    seed_models = {}
    seed_val_accs = {}

    for seed in args.seeds:
        ckpt_path = models_dir / f"scratch_resnet18_seed{seed}.pth"
        if ckpt_path.exists() and not args.force_retrain:
            print(f"\n  Found existing checkpoint for seed {seed}: {ckpt_path}")
            model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.eval()
            seed_models[seed] = model
            # Quick val eval
            val_ds = InMemoryImageDataset(val_samples, transform=get_test_transforms(150))
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
            correct = 0
            with torch.no_grad():
                for imgs, labels, _ in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
            acc = 100.0 * correct / len(val_samples)
            seed_val_accs[seed] = acc
            print(f"  Loaded Seed {seed} | Val Acc: {acc:.2f}%")
        else:
            model, acc = train_single_seed(
                seed=seed,
                train_samples=train_samples,
                val_samples=val_samples,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                save_path=ckpt_path,
            )
            seed_models[seed] = model
            seed_val_accs[seed] = acc

    print("\nEnsemble Validation Baseline:")
    for s, acc in seed_val_accs.items():
        print(f"  Seed {s}: {acc:.2f}%")

    # 3. Run Inference on Unlabeled Pool
    print("\n[3/5] Running inference on 6,000 unlabeled pool images...")
    seed_pool_results = {}
    for seed in args.seeds:
        print(f"  Inference with Seed {seed} model...")
        t0 = time.time()
        seed_pool_results[seed] = run_inference_on_pool(
            seed_models[seed],
            pool_samples,
            batch_size=args.batch_size * 2,
        )
        print(f"  Seed {seed} inference completed in {time.time() - t0:.1f}s")

    # 4. Analyze Agreement & Consensus
    print("\n[4/5] Computing ensemble consensus and uncertainty metrics...")
    pool_records = []

    for img_path, _, row_id in pool_samples:
        preds = [seed_pool_results[s][row_id]["pred"] for s in args.seeds]
        confs = [seed_pool_results[s][row_id]["conf"] for s in args.seeds]
        probs_all = [np.array(seed_pool_results[s][row_id]["probs"]) for s in args.seeds]

        mean_probs = np.mean(probs_all, axis=0)
        mean_pred = int(np.argmax(mean_probs))
        all_agree = (len(set(preds)) == 1)
        min_conf = float(np.min(confs))
        mean_conf = float(np.mean(confs))

        # Margin: difference between top-1 and top-2 mean probability
        sorted_probs = np.sort(mean_probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        # Prediction Entropy
        eps = 1e-8
        entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))

        # Consensus qualification
        is_consensus = all_agree and (min_conf >= args.conf_thresh)

        record = {
            "id": row_id,
            "image": img_path,
            "filename": Path(img_path).name,
            "pred_seed42": preds[0],
            "conf_seed42": confs[0],
            "pred_seed43": preds[1],
            "conf_seed43": confs[1],
            "pred_seed44": preds[2],
            "conf_seed44": confs[2],
            "all_agree": all_agree,
            "agreement_class": preds[0] if all_agree else -1,
            "min_conf": min_conf,
            "mean_conf": mean_conf,
            "margin": margin,
            "entropy": entropy,
            "top1_class": int(np.argsort(mean_probs)[-1]),
            "top2_class": int(np.argsort(mean_probs)[-2]),
            "mean_prob_0": mean_probs[0],
            "mean_prob_1": mean_probs[1],
            "mean_prob_2": mean_probs[2],
            "mean_prob_3": mean_probs[3],
            "mean_prob_4": mean_probs[4],
            "mean_prob_5": mean_probs[5],
            "is_consensus": is_consensus,
        }
        pool_records.append(record)

    df_pool = pd.DataFrame(pool_records)

    # Save full pool analysis
    pool_csv_path = PROJECT_ROOT / "scratch" / "pool_predictions.csv"
    df_pool.to_csv(pool_csv_path, index=False)
    print(f"  [SAVED] All 6,000 pool predictions saved to: {pool_csv_path}")

    # Summary stats
    n_agree = df_pool["all_agree"].sum()
    n_consensus = df_pool["is_consensus"].sum()
    print(f"\nPool Agreement Summary:")
    print(f"  Total Pool:                      {len(df_pool):5d}")
    print(f"  All 3 Seeds Agree:               {n_agree:5d} ({100 * n_agree / len(df_pool):.1f}%)")
    print(f"  Consensus (Agree & Conf >= {args.conf_thresh:.2f}): {n_consensus:5d} ({100 * n_consensus / len(df_pool):.1f}%)")

    for thresh in [0.80, 0.85, 0.90, 0.95]:
        c_count = (df_pool["all_agree"] & (df_pool["min_conf"] >= thresh)).sum()
        print(f"    - Agree & min_conf >= {thresh:.2f}: {c_count:5d} samples")

    # 5. Extract Consensus Dataset
    print("\n[5/5] Extracting Consensus Dataset...")
    df_consensus = df_pool[df_pool["is_consensus"]].copy()

    # If consensus exceeds budget, rank by min_conf descending
    if len(df_consensus) > args.max_consensus:
        print(f"  Consensus samples ({len(df_consensus)}) exceeds max budget ({args.max_consensus}). Selecting top by min_conf...")
        df_consensus = df_consensus.sort_values(by="min_conf", ascending=False).head(args.max_consensus)

    df_consensus["label"] = df_consensus["agreement_class"]
    df_consensus["source"] = "scratch_consensus"
    df_consensus["weight"] = 1.0

    consensus_output_cols = [
        "id", "image", "filename", "label", "weight",
        "pred_seed42", "conf_seed42",
        "pred_seed43", "conf_seed43",
        "pred_seed44", "conf_seed44",
        "min_conf", "mean_conf", "source"
    ]
    df_consensus_out = df_consensus[consensus_output_cols]
    consensus_csv_path = PROJECT_ROOT / "scratch" / "consensus_labels.csv"
    df_consensus_out.to_csv(consensus_csv_path, index=False)
    print(f"  [SAVED] {len(df_consensus_out)} consensus samples saved to: {consensus_csv_path}")

    # Class breakdown
    print("\nConsensus Class Breakdown:")
    for c_idx, c_name in enumerate(CLASSES):
        cnt = (df_consensus_out["label"] == c_idx).sum()
        pct = 100.0 * cnt / len(df_consensus_out) if len(df_consensus_out) > 0 else 0
        print(f"  Class {c_idx:1d} ({c_name:10s}): {cnt:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("  [STEP 1 COMPLETE] Scratch Consensus Pseudo-Labeling finished successfully!")
    print(f"  Seed models saved in: scratch/models/")
    print(f"  Pool predictions:     scratch/pool_predictions.csv")
    print(f"  Consensus labels:     scratch/consensus_labels.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
