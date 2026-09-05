"""
review_cli.py
=============
HackBlox 2026 · 3LC Scene Classification Challenge
Step 3: Interactive CLI for Human Boundary Labeling.

Allows rapid, high-precision manual labeling of the 150-200 hard boundary cases.
Features:
- Live 24-bit ANSI color terminal thumbnail for instantaneous recognition
- Model disagreement signals (predictions & confidences from all 3 seeds)
- Keybindings: 0-5 for classes, 'v' to open full image, 's' to skip, 'u' to undo, 'q' to quit
- Auto-resume: reads existing scratch/human_labels.csv and resumes seamlessly
- Instant disk persistence on every keystroke
"""

import sys
import argparse
from pathlib import Path
import json
import csv
import subprocess
from datetime import datetime
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_DESCS = {
    0: "buildings (houses, architecture, urban skyline)",
    1: "forest    (trees, greenery, foliage)",
    2: "glacier   (ice, snow pack, icebergs)",
    3: "mountain  (rocky peaks, bare slopes)",
    4: "sea       (ocean, waves, water horizon)",
    5: "street    (roads, asphalt, pavement, vehicles)",
}


def render_terminal_thumbnail(image_path: str, width: int = 44, height: int = 16) -> str:
    """Render a 24-bit ANSI half-block color thumbnail in the terminal."""
    try:
        img = Image.open(image_path).convert("RGB").resize((width, height * 2), Image.Resampling.BILINEAR)
        pixels = img.load()
        lines = []
        for y in range(0, height * 2, 2):
            line = []
            for x in range(width):
                r1, g1, b1 = pixels[x, y]
                r2, g2, b2 = pixels[x, y + 1] if y + 1 < height * 2 else (0, 0, 0)
                line.append(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀\033[0m")
            lines.append("".join(line))
        return "\n".join(lines)
    except Exception as e:
        return f"[Thumbnail unavailable: {e}]"


def open_in_viewer(image_path: str):
    """Open full-resolution image in system viewer."""
    viewers = ["xdg-open", "feh", "display", "eog", "open"]
    for v in viewers:
        try:
            subprocess.Popen([v, image_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  [VIEWER] Opened with {v}: {image_path}")
            return
        except FileNotFoundError:
            continue
    print(f"  [WARN] No graphical image viewer found. Image path: {image_path}")


def load_existing_labels(csv_path: Path) -> dict:
    labels = {}
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row["id"])] = {
                    "label": int(row["label"]),
                    "label_name": row["label_name"],
                    "image": row["image"],
                    "pair_type": row.get("pair_type", "unknown"),
                    "timestamp": row.get("timestamp", ""),
                }
    return labels


def save_labels(csv_path: Path, labels_dict: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "image", "label", "label_name", "pair_type", "weight", "source", "reviewed_by", "timestamp"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rid, item in sorted(labels_dict.items()):
            writer.writerow({
                "id": rid,
                "image": item["image"],
                "label": item["label"],
                "label_name": item["label_name"],
                "pair_type": item.get("pair_type", ""),
                "weight": 1.0,
                "source": "human_reviewed",
                "reviewed_by": "expert_human",
                "timestamp": item.get("timestamp", datetime.now().isoformat()),
            })


def main():
    parser = argparse.ArgumentParser(description="HackBlox Human Label Review CLI")
    parser.add_argument("--queue", type=str, default="scratch/hard_cases_queue.json", help="Hard cases queue JSON")
    parser.add_argument("--output", type=str, default="scratch/human_labels.csv", help="Output human labels CSV")
    parser.add_argument("--no-thumb", action="store_true", help="Disable terminal ANSI thumbnail rendering")
    parser.add_argument("--auto-view", action="store_true", help="Automatically launch system image viewer on each sample")
    parser.add_argument("--mock-auto", action="store_true", help="Batch non-interactive test mode (resolves via seed majority / top-1)")
    args = parser.parse_args()

    queue_path = PROJECT_ROOT / args.queue
    csv_path = PROJECT_ROOT / args.output

    if not queue_path.exists():
        print(f"[ERROR] Queue file not found: {queue_path}. Run select_hard_cases.py first!")
        sys.exit(1)

    with open(queue_path, "r") as f:
        queue = json.load(f)

    existing_labels = load_existing_labels(csv_path)
    print("=" * 70)
    print("  HackBlox 2026 · Human Hard-Case Review Tool")
    print(f"  Total Queue: {len(queue)} | Already Reviewed: {len(existing_labels)}")
    print("=" * 70)

    # Mock mode for testing
    if args.mock_auto:
        print("\n[MOCK MODE] Auto-resolving remaining hard cases for pipeline validation...")
        for item in queue:
            rid = item["id"]
            if rid in existing_labels:
                continue
            # Pick top-1 class from ensemble
            resolved_label = item["top1_class"]
            existing_labels[rid] = {
                "label": resolved_label,
                "label_name": CLASSES[resolved_label],
                "image": item["image_path"],
                "pair_type": item["pair_type"],
                "timestamp": datetime.now().isoformat(),
            }
        save_labels(csv_path, existing_labels)
        print(f"[MOCK COMPLETE] Saved {len(existing_labels)} reviews to {csv_path}")
        return

    # Interactive Loop
    history = []
    unreviewed = [item for item in queue if item["id"] not in existing_labels]

    if not unreviewed:
        print("\n  [ALL DONE] All samples in the queue have already been reviewed!")
        print(f"  Labels saved at: {csv_path}")
        return

    idx = 0
    while idx < len(unreviewed):
        item = unreviewed[idx]
        rid = item["id"]
        img_path = item["image_path"]

        # Class counts so far
        counts = {c: 0 for c in range(6)}
        for lbl in existing_labels.values():
            counts[lbl["label"]] += 1

        print("\n" + "-" * 70)
        print(f"  Sample [{len(existing_labels) + 1}/{len(queue)}] | Remaining: {len(unreviewed) - idx} | Progress: {100 * len(existing_labels) / len(queue):.1f}%")
        print(f"  Counts so far: Bld:{counts[0]} | For:{counts[1]} | Gla:{counts[2]} | Mtn:{counts[3]} | Sea:{counts[4]} | Str:{counts[5]}")
        print("-" * 70)

        if not args.no_thumb:
            print(render_terminal_thumbnail(img_path))

        if args.auto_view:
            open_in_viewer(img_path)

        pair_type = item.get("pair_type", f"{item.get('predicted_class', '')} vs {item.get('top2_class', '')}")
        entropy = item.get("entropy", 0.0)
        margin = item.get("margin", 0.0)
        print(f"\n  Image File: {item.get('filename', Path(img_path).name)}")
        print(f"  Image Path: {img_path}")
        print(f"  Boundary / Category: \033[1m{pair_type.upper()}\033[0m (Margin: {margin:.3f}, Entropy: {entropy:.3f})")
        if "pred_seed42" in item:
            print(f"  Ensemble Predictions:")
            print(f"    - Seed 42: {CLASSES[item['pred_seed42']]:8s} (conf: {item.get('conf_seed42', 0):.3f})")
            print(f"    - Seed 43: {CLASSES[item['pred_seed43']]:8s} (conf: {item.get('conf_seed43', 0):.3f})")
            print(f"    - Seed 44: {CLASSES[item['pred_seed44']]:8s} (conf: {item.get('conf_seed44', 0):.3f})")
        t1_name = item.get("top1_name", item.get("predicted_class", "N/A"))
        t2_name = item.get("top2_name", item.get("top2_class", "N/A"))
        t1_cls = item.get("top1_class", item.get("predicted_class_idx", 0))
        t2_cls = item.get("top2_class", item.get("top2_class_idx", 1))
        print(f"  Top-2 Consensus Candidates: \033[33m[{t1_cls}] {t1_name}\033[0m vs \033[36m[{t2_cls}] {t2_name}\033[0m")

        print("\n  Classes:  [0] Buildings   [1] Forest   [2] Glacier   [3] Mountain   [4] Sea   [5] Street")
        print("  Controls: [s] Skip        [v] Open Viewer   [u] Undo Last   [q] Save & Quit")

        while True:
            try:
                choice = input("\n  Enter class [0-5] or control: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n\n  Exiting and saving work...")
                save_labels(csv_path, existing_labels)
                return

            if choice in ["0", "1", "2", "3", "4", "5"]:
                label_val = int(choice)
                existing_labels[rid] = {
                    "label": label_val,
                    "label_name": CLASSES[label_val],
                    "image": img_path,
                    "pair_type": item["pair_type"],
                    "timestamp": datetime.now().isoformat(),
                }
                history.append(rid)
                save_labels(csv_path, existing_labels)
                print(f"  --> Recorded: \033[32m[{label_val}] {CLASSES[label_val]}\033[0m (saved to {csv_path.name})")
                idx += 1
                break

            elif choice == "s":
                print("  --> Skipped.")
                idx += 1
                break

            elif choice == "v":
                open_in_viewer(img_path)

            elif choice == "u":
                if history:
                    last_id = history.pop()
                    if last_id in existing_labels:
                        del existing_labels[last_id]
                        save_labels(csv_path, existing_labels)
                        print(f"  --> Undid label for row ID {last_id}.")
                        # Reposition to previous sample
                        idx = max(0, idx - 1)
                        break
                else:
                    print("  --> Nothing to undo.")

            elif choice == "q":
                print("\n  Saving progress and exiting...")
                save_labels(csv_path, existing_labels)
                print(f"  Saved {len(existing_labels)} reviews to {csv_path}")
                return

            else:
                print("  Invalid input! Enter 0-5, s, v, u, or q.")

    save_labels(csv_path, existing_labels)
    print("\n" + "=" * 70)
    print(f"  [REVIEW COMPLETE] Finished reviewing {len(existing_labels)} hard boundary cases!")
    print(f"  Final review saved to: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
