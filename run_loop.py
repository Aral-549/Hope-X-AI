"""
HackBlox 2026 · 3LC Scene Classification Challenge
Execution Protocol: Iterative Data-Centric Labeling Loop Runner (Loops 1, 2, 3)

Follows strict 10-step protocol:
1. Inspect previous run's embeddings & predictions in 3LC.
2. Select active learning batch according to loop strategy:
   - Loop 1: Uncertainty / Margin Sampling + UMAP Embedding Diversity (+800 samples -> 1,400 active)
   - Loop 2: Class-Balanced Error-Focused Sampling targeting confused classes (+800 samples -> 2,200 active)
   - Loop 3: Strict Hard-Negative Mining Pass (+600 samples -> 2,800 active <= 3,000)
3. Annotate selected samples with high-fidelity vision annotator (mimicking 3LC Dashboard visual inspection).
4. Verify total weight=1 count <= 3,000.
5. Create and save new versioned 3LC table revision (train_0000, train_0001, train_0002) with lineage.
6. Train ResNet-18 strictly from scratch with advanced augmentations and cosine warmup.
7. Collect 3LC metrics and 3D UMAP embeddings.
8. Save proof artifacts: confusion matrix, classification report, 3D embeddings plot.
9. Generate test predictions via predict.py -> submission.csv.
10. Update loop_log.md and metrics.csv.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import tlc

from src.utils import load_config, set_seed
from src.active_learning import (
    get_latest_run_parquet,
    load_pool_dataframe,
    select_loop1_uncertainty_diversity,
    select_loop2_error_focused,
    select_loop3_hard_negatives,
    annotate_samples,
    create_revision,
)

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def parse_args():
    parser = argparse.ArgumentParser(description="Execute 3LC Data-Centric Labeling Loop")
    parser.add_argument("--loop", type=int, required=True, choices=[1, 2, 3], help="Loop index to execute (1, 2, or 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    set_seed(config["project"]["random_seed"])

    loop_idx = args.loop
    print("=" * 75)
    print(f"  EXECUTING PHASE 2 · LOOP {loop_idx} (3LC Data-Centric Pipeline)")
    print("=" * 75)

    # 1. Load latest train table
    print("\n[Step 1/10] Loading current train table revision (.latest())...")
    train_table = tlc.Table.from_names(
        project_name=config["project"]["name"],
        dataset_name=config["project"]["dataset_name"],
        table_name="train",
    ).latest()
    
    current_active = sum(1 for row in train_table.table_rows if row["weight"] > 0)
    print(f"  Current Train Table: {train_table.url}")
    print(f"  Current Active Samples (weight > 0): {current_active} / {config['data']['max_weight1_rows']}")

    # 2. Locate latest run's reduced embeddings
    print("\n[Step 2/10] Locating latest 3LC Run embeddings & metrics...")
    parquet_path, run_name = get_latest_run_parquet(config["project"]["name"])
    print(f"  Using 3LC Run: {run_name}")
    print(f"  Parquet file: {parquet_path}")

    # 3. Load merged pool dataframe
    print("\n[Step 3/10] Loading undefined pool with 3D UMAP coordinates & confidence...")
    df_pool = load_pool_dataframe(train_table, parquet_path)
    print(f"  Total samples in pool table: {len(df_pool)}")
    
    # 4. Active Learning Selection
    print(f"\n[Step 4/10] Running Active Learning Selection Strategy for Loop {loop_idx}...")
    if loop_idx == 1:
        n_to_add = 800
        desc = "Loop 1: Uncertainty / Margin Sampling with UMAP Embedding Diversity"
        selected_ids = select_loop1_uncertainty_diversity(df_pool, n_to_add=n_to_add, seed=config["project"]["random_seed"])
    elif loop_idx == 2:
        n_to_add = 800
        desc = "Loop 2: Class-Balanced Error-Focused Sampling (Glacier/Mountain/Buildings/Street)"
        selected_ids = select_loop2_error_focused(df_pool, n_to_add=n_to_add, seed=config["project"]["random_seed"])
    elif loop_idx == 3:
        n_to_add = 600
        desc = "Loop 3: Hard-Negative Mining Pass (Boundary Ambiguity Pairs)"
        selected_ids = select_loop3_hard_negatives(df_pool, n_to_add=n_to_add, seed=config["project"]["random_seed"])

    print(f"  Selected {len(selected_ids)} new samples to activate.")
    df_selected = df_pool[df_pool["id"].isin(selected_ids)].copy()

    # 5. Label verification / Annotation
    print(f"\n[Step 5/10] Annotating {len(df_selected)} selected samples (mimicking 3LC Dashboard visual curation)...")
    annotations = annotate_samples(df_selected)
    
    # Show annotation distribution
    labels_assigned = [v["label"] for v in annotations.values()]
    print("  Curated Label Distribution:")
    for c_i, c_name in enumerate(CLASSES):
        count = sum(1 for l in labels_assigned if l == c_i)
        print(f"    {c_i} ({c_name:10s}): {count:4d} samples")

    # 6. Verify total budget before writing
    new_active_total = current_active + len(selected_ids)
    print(f"\n[Step 6/10] Verifying Labeling Budget: {new_active_total} / {config['data']['max_weight1_rows']}...")
    if new_active_total > config["data"]["max_weight1_rows"]:
        raise ValueError(f"FATAL: Total active rows ({new_active_total}) exceeds budget ({config['data']['max_weight1_rows']})!")

    # 7. Create new 3LC Table Revision
    print(f"\n[Step 7/10] Creating and finalizing versioned 3LC Table Revision...")
    new_train_table, active_count = create_revision(train_table, annotations, loop_idx=loop_idx, description=desc)
    print(f"  [OK] New Revision Table URL: {new_train_table.url}")
    print(f"  [OK] Confirmed active rows (weight=1.0): {active_count}")

    # 8. Train Model on New Table Revision
    print(f"\n[Step 8/10] Training ResNet-18 from scratch on new table revision...")
    import subprocess
    train_cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "train.py"),
        "--loop", str(loop_idx),
        "--advanced",
    ]
    res = subprocess.run(train_cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Training failed with return code {res.returncode}")

    # 9. Generate Test Set Predictions
    print(f"\n[Step 9/10] Generating test set predictions & submission.csv...")
    predict_cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "predict.py"),
    ]
    res = subprocess.run(predict_cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Prediction failed with return code {res.returncode}")

    # 10. Update Loop Log
    print(f"\n[Step 10/10] Updating logs/loop_log.md with Loop {loop_idx} audit record...")
    # Read latest validation accuracy from classification report
    report_file = PROJECT_ROOT / "reports" / f"loop{loop_idx}" / "confusion_matrix_report.txt"
    report_text = report_file.read_text() if report_file.exists() else "Report not found"
    
    # Read previous validation accuracy
    metrics_file = PROJECT_ROOT / "logs" / "metrics.csv"
    metrics_df = pd.read_csv(metrics_file)
    val_acc_now = float(metrics_df.iloc[-1]["val_accuracy"])
    val_acc_prev = float(metrics_df.iloc[-2]["val_accuracy"]) if len(metrics_df) > 1 else 0.0

    log_entry = f"""
### Loop {loop_idx} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
- **Strategy / Criterion**: {desc}
- **Samples Touched / Newly Curated**: {len(selected_ids)} samples
- **Total Active Training Samples (weight = 1.0)**: {active_count} / {config['data']['max_weight1_rows']}
- **Table Revision**: `{new_train_table.url}` (parent: `{train_table.url}`)
- **Validation Accuracy**: {val_acc_prev}% → **{val_acc_now:.2f}%** ({'+' if val_acc_now >= float(val_acc_prev) else ''}{val_acc_now - float(val_acc_prev):.2f}%)
- **Confusion Shifts & Key Findings**:
  - Activated {len(selected_ids)} curated samples targeting decision boundary uncertainty.
  - Proof artifacts generated: `reports/loop{loop_idx}/confusion_matrix.png`, `confusion_matrix.csv`, `confusion_matrix_report.txt`, `embedding_view.png`.
  - Predictions generated and formatted to `submission.csv` and archived to `submissions/`.
"""
    with open(PROJECT_ROOT / "logs" / "loop_log.md", "a") as f:
        f.write(log_entry)

    print("=" * 75)
    print(f"  [SUCCESS] LOOP {loop_idx} COMPLETED SUCCESSFULLY!")
    print(f"  Validation Accuracy: {val_acc_now:.2f}%")
    print(f"  Active Samples: {active_count} / {config['data']['max_weight1_rows']}")
    print(f"  Table Revision: {new_train_table.url}")
    print("=" * 75)


if __name__ == "__main__":
    main()
