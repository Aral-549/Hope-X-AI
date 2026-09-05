"""
Train ResNet-18 classifier on Intel Scene dataset using 3LC.
Strictly follows HackBlox 2026 / 3LC Challenge rules:
1. ResNet-18 only, initialized and trained from scratch (weights=None).
2. Max 3,000 active samples (weight=1.0) strictly enforced before training.
3. Versioned 3LC lineage tracking with input_tables and .latest() resolution.
4. Generates all proof artifacts: confusion matrix, report, 3D embeddings, metrics.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tlc

from src.utils import set_seed, load_config, plot_and_save_confusion_matrix
from src.model import ResNet18Classifier, verify_from_scratch
from src.augment import get_baseline_transforms, get_advanced_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3LC Scene Classifier")
    parser.add_argument("--loop", type=int, default=0, help="Current loop index (0=baseline, 1, 2, 3)")
    parser.add_argument("--table-url", type=str, default=None, help="Explicit 3LC table URL to pin revision")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--save-name", type=str, default=None, help="Custom checkpoint filename")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip 3LC metric collection & UMAP reduction")
    parser.add_argument("--advanced", action="store_true", help="Use advanced augmentation & regularization recipe")
    return parser.parse_args()


def metrics_fn(batch, predictor_output: tlc.PredictorOutput):
    labels = batch[1].to(device)
    predictions = predictor_output.forward
    softmax_output = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1)
    confidence = torch.gather(softmax_output, 1, predicted_indices.unsqueeze(1)).squeeze(1)
    accuracy = (predicted_indices == labels).float()

    valid_labels = labels < predictions.shape[1]
    cross_entropy_loss = torch.ones_like(labels, dtype=torch.float32)
    cross_entropy_loss[valid_labels] = nn.CrossEntropyLoss(reduction="none")(
        predictions[valid_labels], labels[valid_labels]
    )
    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }


def save_embedding_plot(parquet_path: Path, output_path: Path, title_suffix: str = ""):
    """Generate high-resolution side-by-side 3D UMAP embedding visualization."""
    df = pd.read_parquet(parquet_path)
    coords = np.array(df["embeddings_mean_67_umap"].tolist())

    fig = plt.figure(figsize=(16, 7))

    # 1. 3D Plot Colored by Predicted Label
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    colors = ["#e41a1c", "#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#a65628"]

    for i, cls_name in enumerate(classes):
        mask = df["predicted"] == i
        ax1.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=colors[i], label=cls_name, s=8, alpha=0.6
        )
    ax1.set_title(f"3LC UMAP 3D Latent Embeddings (Predicted Class) {title_suffix}", fontsize=12, pad=10)
    ax1.legend(loc="upper right", markerscale=3)
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_zlabel("UMAP 3")

    # 2. 3D Plot Colored by Confidence
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc = ax2.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=df["confidence"], cmap="viridis", s=8, alpha=0.6, vmin=0.2, vmax=1.0
    )
    cbar = fig.colorbar(sc, ax=ax2, shrink=0.6, aspect=15)
    cbar.set_label("Prediction Confidence", fontsize=10)
    ax2.set_title(f"3LC UMAP 3D Latent Embeddings (Confidence) {title_suffix}", fontsize=12, pad=10)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.set_zlabel("UMAP 3")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[REPORT] Saved embedding visualization to: {output_path}")


def main():
    args = parse_args()
    config = load_config()

    seed = args.seed if args.seed is not None else config["project"]["random_seed"]
    set_seed(seed)

    print("=" * 70)
    print(f"  HackBlox 2026 · 3LC Scene Classification (Loop {args.loop})")
    print(f"  Device: {device} | Architecture: ResNet-18 (Random init, from scratch)")
    print("=" * 70)

    tlc.register_project_url_alias(
        token="INTEL_SCENE_DATA",
        path=str(PROJECT_ROOT.absolute()),
        project=config["project"]["name"],
    )

    # 1. Load Tables
    print("\n[1/6] Loading 3LC Tables...")
    if args.table_url:
        print(f"  Loading train table by URL: {args.table_url}")
        train_table = tlc.Table.from_url(args.table_url)
    else:
        print("  Loading train table by name (.latest())...")
        train_table = tlc.Table.from_names(
            project_name=config["project"]["name"],
            dataset_name=config["project"]["dataset_name"],
            table_name="train",
        ).latest()

    val_table = tlc.Table.from_names(
        project_name=config["project"]["name"],
        dataset_name=config["project"]["dataset_name"],
        table_name="val",
    ).latest()

    print(f"  Train table rows: {len(train_table)} | URL: {train_table.url}")
    print(f"  Val table rows:   {len(val_table)} | URL: {val_table.url}")

    # 2. Hard Budget Constraint Check
    max_rows = config["data"]["max_weight1_rows"]
    n_weight1 = sum(1 for row in train_table.table_rows if row["weight"] > 0)
    print(f"\n[2/6] Labeling Budget Check: {n_weight1} / {max_rows} active (weight > 0) rows")
    if n_weight1 > max_rows:
        print(f"\n[FATAL ERROR] Labeling budget exceeded: {n_weight1} > {max_rows}!")
        print("  Competition rules strictly forbid training tables exceeding 3,000 weight=1 samples.")
        sys.exit(1)

    # 3. Setup transforms & dataloaders
    cfg_train = config["training"]["advanced" if args.advanced else "baseline"]
    epochs = args.epochs or cfg_train["epochs"]
    lr = args.lr or cfg_train["learning_rate"]
    batch_size = cfg_train["batch_size"]

    if args.advanced:
        train_tf, val_tf = get_advanced_transforms(config["data"]["image_size"])
    else:
        train_tf, val_tf = get_baseline_transforms(config["data"]["image_size"])

    def train_fn(sample):
        img = Image.open(sample["image"]).convert("RGB")
        return train_tf(img), sample["label"]

    def val_fn(sample):
        img = Image.open(sample["image"]).convert("RGB")
        return val_tf(img), sample["label"]

    train_table.map(train_fn).map_collect_metrics(val_fn)
    val_table.map(val_fn)

    train_sampler = train_table.create_sampler(exclude_zero_weights=True)
    train_loader = DataLoader(
        train_table,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(val_table, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Model, Criterion, Optimizer, Scheduler
    print("\n[3/6] Initializing Model from scratch...")
    model = ResNet18Classifier(
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"],
    ).to(device)
    verify_from_scratch(model)

    label_smoothing = cfg_train.get("label_smoothing", 0.1) if args.advanced else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg_train.get("weight_decay", 1e-4))

    if args.advanced and cfg_train.get("scheduler") == "cosine_warmup":
        warmup_epochs = cfg_train.get("warmup_epochs", 3)
        min_lr = cfg_train.get("min_lr", 1e-6)
        def lr_lambda(epoch_idx):
            if epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(max(1, warmup_epochs))
            progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return max(min_lr / lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg_train.get("step_size", 5), gamma=cfg_train.get("gamma", 0.1))

    # 5. Initialize 3LC Run
    class_names = list(train_table.get_simple_value_map("label").values())
    run = tlc.init(
        project_name=config["project"]["name"],
        description=f"Intel Scene - Loop {args.loop} ({'Advanced' if args.advanced else 'Baseline'})",
    )
    metric_schemas = {
        "loss": tlc.Schema(description="Cross entropy loss", value=tlc.Float32Value()),
        "predicted": tlc.CategoricalLabelSchema(display_name="predicted label", classes=class_names),
        "accuracy": tlc.Schema(description="Per-sample accuracy", value=tlc.Float32Value()),
        "confidence": tlc.Schema(description="Prediction confidence", value=tlc.Float32Value()),
    }
    classification_metrics_collector = tlc.FunctionalMetricsCollector(
        collection_fn=metrics_fn,
        column_schemas=metric_schemas,
    )
    indices_and_modules = list(enumerate(model.resnet.named_modules()))
    resnet_fc_layer_index = next((i for i, (n, _) in indices_and_modules if n == "fc"), len(indices_and_modules) - 1)
    embeddings_metrics_collector = tlc.EmbeddingsMetricsCollector(layers=[resnet_fc_layer_index])
    predictor = tlc.Predictor(model, layers=[resnet_fc_layer_index])

    # 6. Training Loop
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0

    print(f"\n[4/6] Training for {epochs} epochs (lr={lr}, batch_size={batch_size}, active_samples={n_weight1})...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(labels.cpu().numpy())

        val_accuracy = 100.0 * val_correct / val_total
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{epochs} (lr={current_lr:.6f}) | Val Acc: {val_accuracy:.2f}% (Best: {best_val_accuracy:.2f}%)")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            best_val_preds = list(all_val_preds)
            best_val_targets = list(all_val_targets)
            print("  --> [CHECKPOINT] New best validation score!")

        tlc.log({"epoch": epoch, "val_accuracy": val_accuracy})

    print("\n" + "=" * 70)
    print(f"  [RESULT] Best Validation Accuracy: {best_val_accuracy:.2f}% at Epoch {best_epoch}")
    print("=" * 70)

    # Save checkpoints
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if args.save_name:
        custom_ckpt = PROJECT_ROOT / args.save_name
        torch.save(model.state_dict(), custom_ckpt)
        print(f"[OK] Custom checkpoint saved to {custom_ckpt}")
    else:
        ckpt_path = PROJECT_ROOT / config["data"]["paths"]["best_model_path"]
        torch.save(model.state_dict(), ckpt_path)
        loop_ckpt_path = PROJECT_ROOT / f"best_model_loop{args.loop}.pth"
        torch.save(model.state_dict(), loop_ckpt_path)
        print(f"[OK] Checkpoints saved to {ckpt_path} and {loop_ckpt_path}")

        # Generate and save confusion matrix
        loop_folder = "loop0_baseline" if args.loop == 0 else f"loop{args.loop}"
        report_dir = PROJECT_ROOT / config["data"]["paths"]["reports_dir"] / loop_folder
        cm_path = report_dir / "confusion_matrix.png"
        target_names = config["data"]["classes"]
        plot_and_save_confusion_matrix(
            best_val_targets,
            best_val_preds,
            class_names=target_names,
            save_path=str(cm_path),
            title=f"Confusion Matrix - Loop {args.loop} (Val Acc: {best_val_accuracy:.2f}%)",
        )

    if not args.skip_metrics:
        # 7. Collect 3LC Metrics & Embeddings
        print("\n[5/6] Collecting per-sample metrics on train table...")
        model.eval()
        tlc.collect_metrics(
            train_table,
            predictor=predictor,
            metrics_collectors=[classification_metrics_collector, embeddings_metrics_collector],
            split="train",
            dataloader_args={"batch_size": batch_size, "num_workers": 0},
        )

        print("\n[6/6] Reducing embeddings with UMAP (3D)...")
        try:
            reduced_meta = run.reduce_embeddings_by_foreign_table_url(
                train_table.url,
                method="umap",
                n_neighbors=15,
                n_components=3,
            )
            print("  [OK] Embeddings successfully reduced to 3D UMAP space.")
            
            # Save side-by-side 3D embedding visualization
            run_name = run.name
            parquet_file = Path.home() / ".local/share/3LC/projects" / config["project"]["name"] / "runs" / run_name / "reduced_0000" / "reduced_0000.parquet"
            if parquet_file.exists():
                emb_plot_path = report_dir / "embedding_view.png"
                save_embedding_plot(parquet_file, emb_plot_path, title_suffix=f"- Loop {args.loop}")
        except Exception as e:
            print(f"  [WARN] UMAP reduction / visualization exception: {e}")

    run.set_status_completed()

    # Append to metrics.csv
    logged_ckpt = custom_ckpt if args.save_name else ckpt_path
    metrics_file = PROJECT_ROOT / config["data"]["paths"]["logs_dir"] / "metrics.csv"
    timestamp = datetime.now().isoformat()
    with open(metrics_file, "a") as f:
        f.write(f"{args.loop},{timestamp},{epochs},{batch_size},{lr},{n_weight1},NA,{best_val_accuracy:.4f},NA,{train_table.url},{logged_ckpt}\n")

    print(f"\n[OK] Run registered in 3LC. Table URL: {train_table.url}")
    print("=" * 70)


if __name__ == "__main__":
    main()
