"""
build_table.py
==============
HackBlox 2026 · 3LC Scene Classification Challenge
Step 4: Merge, Guardrail Verification & 3LC Table Registration.

Merges:
1. Base clean seed samples (600 official samples, weight=1.0)
2. Scratch consensus pseudo-labeled samples (from scratch/consensus_labels.csv, weight=1.0)
3. Human-reviewed boundary samples (from scratch/human_labels.csv, weight=1.0)
4. Remaining unselected pool images (weight=0.0, label=6 "undefined")

Enforces strict competition rules:
- Active samples (weight=1.0) <= 3,000
- Lineage anchored directly to root clean table 'train'
- Zero CLIP or external model dependencies
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
ALL_CLASSES = CLASSES + ["undefined"]

SCHEMAS = {
    "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
    "image": tlc.ImagePath,
    "label": tlc.CategoricalLabel("label", classes=ALL_CLASSES),
    "weight": tlc.SampleWeightSchema(),
}


def main():
    parser = argparse.ArgumentParser(description="Build and Register Clean 3LC Dataset Revision")
    parser.add_argument("--consensus-csv", type=str, default="scratch/consensus_labels.csv", help="Consensus CSV path")
    parser.add_argument("--human-csv", type=str, default="scratch/human_labels_clean.csv", help="Human labels CSV path")
    parser.add_argument("--round2-csv", type=str, default="scratch/round2_consensus_labels.csv", help="Round 2 Consensus CSV path")
    parser.add_argument("--table-name", type=str, default="train", help="Table name in 3LC (will append revision)")
    parser.add_argument("--allow-no-human", action="store_true", help="Allow building without human labels")
    args = parser.parse_args()

    config = load_config()
    max_active = config["data"]["max_weight1_rows"]

    print("=" * 70)
    print("  HackBlox 2026 · 3LC Clean Table Builder & Guardrail Verifier")
    print("=" * 70)

    # 1. Load Root Base Table
    root_table_url = "/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train"
    print(f"\n[1/4] Loading base root table: {root_table_url}")
    root_table = tlc.Table.from_url(root_table_url)
    print(f"  Root table total rows: {len(root_table)}")

    # 2. Collect Annotations
    new_annotations = {}

    # Load Round 1 consensus
    consensus_path = PROJECT_ROOT / args.consensus_csv
    if consensus_path.exists():
        df_consensus = pd.read_csv(consensus_path)
        for _, row in df_consensus.iterrows():
            rid = int(row["id"])
            new_annotations[rid] = {
                "label": int(row["label"]),
                "source": "scratch_consensus",
            }
        print(f"  [CONSENSUS R1] Loaded {len(df_consensus)} labels from {consensus_path.name}")
    else:
        print(f"  [WARN] No consensus CSV found at {consensus_path}")

    # Load Round 2 consensus
    if args.round2_csv:
        r2_path = PROJECT_ROOT / args.round2_csv
        if r2_path.exists():
            df_r2 = pd.read_csv(r2_path)
            for _, row in df_r2.iterrows():
                rid = int(row["id"])
                new_annotations[rid] = {
                    "label": int(row["label"]),
                    "source": "scratch_consensus_round2",
                }
            print(f"  [CONSENSUS R2] Loaded {len(df_r2)} labels from {r2_path.name}")

    # Load human labels (corroborated)
    human_path = PROJECT_ROOT / args.human_csv
    if human_path.exists():
        df_human = pd.read_csv(human_path)
        for _, row in df_human.iterrows():
            rid = int(row["id"])
            # Human labels take highest precedence if overlap
            new_annotations[rid] = {
                "label": int(row["label"]),
                "source": "human_reviewed",
            }
        print(f"  [HUMAN] Loaded {len(df_human)} expert labels from {human_path.name}")
    else:
        if not args.allow_no_human:
            print(f"  [WARN] Human labels not found at {human_path}. Run review_cli.py or use --allow-no-human.")

    print(f"  Total newly activated samples to apply: {len(new_annotations)}")

    # 3. Build New Revision
    print(f"\n[2/4] Assembling Table Rows & Enforcing Guardrails...")
    writer = tlc.TableWriter(
        table_name=args.table_name,
        dataset_name=config["project"]["dataset_name"],
        project_name=config["project"]["name"],
        description=f"Intel Scene Clean Revision (Consensus: {len(df_consensus if consensus_path.exists() else [])}, Human: {len(df_human if human_path.exists() else [])})",
        column_schemas=SCHEMAS,
        input_tables=[root_table.url],
        if_exists="rename",
    )

    active_count = 0
    updated_count = 0
    base_seed_count = 0
    remaining_undefined = 0
    class_distribution = {c: 0 for c in range(6)}

    for row in root_table.table_rows:
        row_dict = dict(row)
        row_id = row_dict["id"]

        if row_id in new_annotations:
            ann = new_annotations[row_id]
            label = ann["label"]
            weight = 1.0
            updated_count += 1
            active_count += 1
            class_distribution[label] += 1
        else:
            label = row_dict["label"]
            weight = float(row_dict.get("weight", 0.0))
            if weight > 0:
                active_count += 1
                base_seed_count += 1
                class_distribution[label] += 1
            else:
                remaining_undefined += 1

        writer.add_row({
            "id": row_id,
            "image": row_dict["image"],
            "label": label,
            "weight": weight,
        })

    # Strict budget validation
    print(f"\n[3/4] Budget Verification:")
    print(f"  Base Seed samples (weight=1.0):      {base_seed_count:5d}")
    print(f"  Newly activated samples (weight=1.0): {updated_count:5d}")
    print(f"  Total Active Samples:                {active_count:5d} / {max_active} (Max Limit)")
    print(f"  Remaining Undefined Pool:            {remaining_undefined:5d}")
    print(f"  Total Table Rows:                    {active_count + remaining_undefined:5d}")

    if active_count > max_active:
        raise ValueError(
            f"FATAL: Labeling budget violated! Active count {active_count} exceeds competition limit {max_active}!"
        )

    print("\nActive Class Distribution:")
    for c_idx, c_name in enumerate(CLASSES):
        print(f"  Class {c_idx} ({c_name:10s}): {class_distribution[c_idx]:5d} samples")

    print("\n[4/4] Finalizing 3LC Table Registration...")
    new_table = writer.finalize()
    print(f"  [OK] Successfully finalized clean revision!")
    print(f"  Table URL:        {new_table.url}")
    print(f"  Parent Lineage:   {root_table.url}")

    # Export metadata
    meta = {
        "table_name": args.table_name,
        "table_url": str(new_table.url),
        "parent_table_url": str(root_table.url),
        "total_rows": len(new_table),
        "active_weight1_rows": active_count,
        "base_seed_rows": base_seed_count,
        "consensus_rows": len(df_consensus) if consensus_path.exists() else 0,
        "human_rows": len(df_human) if human_path.exists() else 0,
        "class_distribution": {CLASSES[k]: v for k, v in class_distribution.items()},
        "compliance": {
            "no_clip": True,
            "scratch_only": True,
            "budget_compliant": True,
        }
    }
    meta_path = PROJECT_ROOT / "3lc_tables" / "train_clean_revision_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [SAVED] Metadata saved to: {meta_path}")

    print("\n" + "=" * 70)
    print(f"  [SUCCESS] Clean table ready for training at: {new_table.url}")
    print("=" * 70)
    return str(new_table.url)


if __name__ == "__main__":
    main()
