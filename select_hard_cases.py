"""
select_hard_cases.py
====================
HackBlox 2026 · 3LC Scene Classification Challenge
Step 2: Selection of High-Ambiguity Boundary Cases for Human Review.

Identifies the most informative samples from the remaining unlabeled pool:
- High ensemble disagreement (seeds predicting different classes)
- Low classification margin between top-2 predicted classes
- Proximity to known confusion boundaries:
    * glacier <-> mountain
    * glacier <-> sea
    * buildings <-> street
    * forest <-> mountain

Outputs:
- scratch/hard_cases_queue.json
- scratch/hard_cases_queue.csv
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def get_pair_type(top1_class: int, top2_class: int) -> str:
    top_pair = {int(top1_class), int(top2_class)}
    if top_pair == {2, 3}:
        return "glacier_mountain"
    elif top_pair == {2, 4}:
        return "glacier_sea"
    elif top_pair == {0, 5}:
        return "buildings_street"
    elif top_pair == {1, 3}:
        return "forest_mountain"
    else:
        return "other"


def main():
    parser = argparse.ArgumentParser(description="Select Hard Ambiguity Cases for Human Review")
    parser.add_argument("--pool-csv", type=str, default="scratch/pool_predictions.csv", help="Pool predictions CSV")
    parser.add_argument("--consensus-csv", type=str, default="scratch/consensus_labels.csv", help="Consensus labels CSV")
    parser.add_argument("--total-cases", type=int, default=180, help="Total hard cases to extract")
    parser.add_argument("--out-json", type=str, default="scratch/hard_cases_queue.json", help="Output queue JSON")
    parser.add_argument("--out-csv", type=str, default="scratch/hard_cases_queue.csv", help="Output queue CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("  HackBlox 2026 · Hard Boundary Case Selection Engine")
    print(f"  Target Cases: {args.total_cases}")
    print("=" * 70)

    pool_path = PROJECT_ROOT / args.pool_csv
    consensus_path = PROJECT_ROOT / args.consensus_csv

    if not pool_path.exists():
        raise FileNotFoundError(f"Pool predictions not found at {pool_path}. Run scratch_consensus.py first!")

    df_pool = pd.read_csv(pool_path)
    print(f"  Loaded pool predictions: {len(df_pool)} samples")

    if consensus_path.exists():
        df_consensus = pd.read_csv(consensus_path)
        consensus_ids = set(df_consensus["id"].tolist())
        leftover = df_pool[~df_pool["id"].isin(consensus_ids)].copy()
        print(f"  Excluded {len(consensus_ids)} consensus samples. Leftover pool: {len(leftover)} samples")
    else:
        print("  No consensus CSV found; using all pool samples.")
        leftover = df_pool.copy()

    leftover["pair_type"] = [
        get_pair_type(r["top1_class"], r["top2_class"])
        for _, r in leftover.iterrows()
    ]

    # Calculate ambiguity score
    def calc_ambiguity(row):
        distinct_preds = len(set([row["pred_seed42"], row["pred_seed43"], row["pred_seed44"]]))
        if distinct_preds == 3:
            disagree_val = 1.2
        elif distinct_preds == 2:
            disagree_val = 0.9
        else:
            disagree_val = 1.0 - float(row["min_conf"])

        margin_val = 1.0 - float(row["margin"])
        entropy_val = float(row["entropy"])
        return float(disagree_val + 0.6 * margin_val + 0.3 * entropy_val)

    leftover["ambiguity_score"] = leftover.apply(calc_ambiguity, axis=1)

    # Proportional quotas for balanced coverage of all key confusion pairs
    # Target: ~180 cases
    total = args.total_cases
    quotas = {
        "glacier_mountain": int(round(total * 0.38)),  # ~68
        "glacier_sea": int(round(total * 0.28)),       # ~50
        "buildings_street": int(round(total * 0.28)),  # ~50
        "forest_mountain": int(round(total * 0.06)),   # ~12
    }
    # Adjust for rounding
    diff = total - sum(quotas.values())
    quotas["glacier_mountain"] += diff

    print("\nQuota targets for human review:")
    for k, v in quotas.items():
        print(f"  {k:20s}: {v:3d} samples")

    selected_dfs = []
    for pair_type, quota in quotas.items():
        subset = leftover[leftover["pair_type"] == pair_type].sort_values(
            by="ambiguity_score", ascending=False
        ).head(quota)
        selected_dfs.append(subset)
        print(f"  Selected {len(subset):3d} / {quota:3d} for {pair_type}")

    df_selected = pd.concat(selected_dfs).sort_values(by="ambiguity_score", ascending=False).reset_index(drop=True)
    df_selected["review_order"] = range(1, len(df_selected) + 1)

    # Save to CSV
    out_csv = PROJECT_ROOT / args.out_csv
    df_selected.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Selected hard cases CSV saved to: {out_csv}")

    # Save to JSON for CLI tool
    queue_records = []
    for _, r in df_selected.iterrows():
        queue_records.append({
            "order": int(r["review_order"]),
            "id": int(r["id"]),
            "filename": str(r["filename"]),
            "image_path": str(r["image"]),
            "pair_type": str(r["pair_type"]),
            "ambiguity_score": round(float(r["ambiguity_score"]), 4),
            "margin": round(float(r["margin"]), 4),
            "entropy": round(float(r["entropy"]), 4),
            "pred_seed42": int(r["pred_seed42"]),
            "conf_seed42": round(float(r["conf_seed42"]), 4),
            "pred_seed43": int(r["pred_seed43"]),
            "conf_seed43": round(float(r["conf_seed43"]), 4),
            "pred_seed44": int(r["pred_seed44"]),
            "conf_seed44": round(float(r["conf_seed44"]), 4),
            "top1_class": int(r["top1_class"]),
            "top2_class": int(r["top2_class"]),
            "top1_name": CLASSES[int(r["top1_class"])],
            "top2_name": CLASSES[int(r["top2_class"])],
        })

    out_json = PROJECT_ROOT / args.out_json
    with open(out_json, "w") as f:
        json.dump(queue_records, f, indent=2)
    print(f"[SAVED] Selected hard cases JSON queue saved to: {out_json}")

    print("\n" + "=" * 70)
    print(f"  [STEP 2 COMPLETE] Selected {len(queue_records)} boundary hard cases for human review.")
    print("=" * 70)


if __name__ == "__main__":
    main()
