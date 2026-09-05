import json
from pathlib import Path

parts = ["scratch/reviewed_part1.json", "scratch/reviewed_part2.json", "scratch/reviewed_part3.json", "scratch/reviewed_part4.json"]
all_reviews = {}

for p in parts:
    path = Path(p)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            all_reviews.update({int(k): v for k, v in data.items()})
            print(f"Loaded {len(data)} reviews from {p}")
    else:
        print(f"Missing {p}")

print(f"\nTotal collected reviews: {len(all_reviews)}")
classes = {0: "buildings", 1: "forest", 2: "glacier", 3: "mountain", 4: "sea", 5: "street", -1: "skip"}
counts = {}
for k, v in all_reviews.items():
    lbl = v if isinstance(v, int) else v.get("label", -1)
    name = classes.get(lbl, "unknown")
    counts[name] = counts.get(name, 0) + 1

print("\nClass Breakdown of visually reviewed samples:")
for c_name, count in sorted(counts.items()):
    print(f"  {c_name:12s}: {count:4d}")
