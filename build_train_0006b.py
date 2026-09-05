"""
build_train_0006b.py
====================
Builds 3LC table revision train_0006b:
- Parent: train_0005 (input_tables=['../train_0005'])
- Adds only mountain, forest, glacier, sea additions from final_human_labels.csv (121 samples)
- Zero additions for buildings and street
- train_0006 is left completely intact
"""

import json
from pathlib import Path
import pandas as pd
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
ALL_CLASSES = CLASSES + ["undefined"]

SCHEMAS = {
    "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
    "image": tlc.ImagePath,
    "label": tlc.CategoricalLabel("label", classes=ALL_CLASSES),
    "weight": tlc.SampleWeightSchema(),
}

parent_url = "/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0005"
parent_table = tlc.Table.from_url(parent_url)

df_all = pd.read_csv(PROJECT_ROOT / "scratch/final_human_labels.csv")
allowed = ["mountain", "forest", "glacier", "sea"]
df_filtered = df_all[df_all["label_name"].isin(allowed)].copy()

new_annotations = {int(r["id"]): int(r["label"]) for _, r in df_filtered.iterrows()}

writer = tlc.TableWriter(
    table_name="train_0006b",
    dataset_name="intel-scene",
    project_name="Intel-Scene",
    description="train_0006b: surgical revision keeping only mountain, forest, glacier, sea additions from train_0005",
    column_schemas=SCHEMAS,
    input_tables=[parent_table.url],
    if_exists="overwrite",
)

active_count = 0
updated_count = 0
class_dist = {c: 0 for c in range(6)}

for row in parent_table.table_rows:
    row_dict = dict(row)
    rid = row_dict["id"]
    if rid in new_annotations:
        label = new_annotations[rid]
        weight = 1.0
        updated_count += 1
        active_count += 1
        class_dist[label] += 1
    else:
        label = row_dict["label"]
        weight = float(row_dict.get("weight", 0.0))
        if weight > 0:
            active_count += 1
            class_dist[label] += 1

    writer.add_row({
        "id": rid,
        "image": row_dict["image"],
        "label": label,
        "weight": weight,
    })

new_table = writer.finalize()
print(f"[OK] Successfully built {new_table.url}")
print(f"Parent Table:            {parent_table.url}")
print(f"Newly added rows:        {updated_count} (mountain, forest, glacier, sea ONLY)")
print(f"Total Active Rows:       {active_count} / 3000")
print(f"Total Table Rows:        {len(new_table)}")
print("\nClass Distribution of train_0006b:")
for c_idx, c_name in enumerate(CLASSES):
    print(f"  Class {c_idx} ({c_name:10s}): {class_dist[c_idx]:4d} samples")
