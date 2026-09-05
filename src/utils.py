import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def set_seed(seed: int = 42):
    """
    Ensure complete reproducibility across python, numpy, torch, and CUDA.
    Fixes seeds everywhere and enforces deterministic CUDA behavior.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[REPRODUCIBILITY] All seeds set to {seed} (cudnn.deterministic=True, cudnn.benchmark=False)")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load yaml configuration file."""
    path = Path(config_path)
    if not path.exists():
        # Fallback if called from subfolder
        path = Path(__file__).resolve().parent.parent / config_path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_and_save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path: str,
    title: str = "Confusion Matrix",
):
    """
    Generate, display, and save normalized and raw confusion matrices with detailed metrics.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("True Ground Truth Label", fontsize=12)
    ax.set_xlabel("Predicted Model Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    # Save raw CSV
    csv_path = Path(save_path).with_suffix(".csv")
    import pandas as pd
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(csv_path)

    # Classification report text
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    txt_path = Path(save_path).with_name(Path(save_path).stem + "_report.txt")
    with open(txt_path, "w") as f:
        f.write(f"=== {title} ===\n\n")
        f.write(report)

    print(f"[REPORT] Saved confusion matrix: {save_path}")
    print(f"[REPORT] Saved confusion matrix CSV: {csv_path}")
    print(f"[REPORT] Saved classification metrics: {txt_path}")
    return cm, cm_norm, report
