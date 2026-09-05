"""
generate_review_html.py
=======================
Generates a fast, visual browser-based contact sheet for reviewing the 180 hard cases.
All images are displayed with their top-2 model predictions and radio buttons (0-5).
Features a single button to "Export human_labels.csv" directly into the scratch folder or download.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
queue_path = PROJECT_ROOT / "scratch" / "hard_cases_queue.json"
html_out = PROJECT_ROOT / "scratch" / "review_sheet.html"

with open(queue_path, "r") as f:
    queue = json.load(f)

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HackBlox 2026 - Human Review Tool ({len(queue)} samples)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }}
  h1 {{ font-size: 22px; margin-bottom: 5px; }}
  .header {{ position: sticky; top: 0; background: #1e293b; padding: 15px 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 100; display: flex; justify-content: space-between; align-items: center; }}
  .btn {{ background: #2563eb; color: white; border: none; padding: 10px 20px; font-weight: 600; border-radius: 6px; cursor: pointer; }}
  .btn:hover {{ background: #1d4ed8; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin-top: 20px; }}
  .card {{ background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155; display: flex; flex-direction: column; }}
  .card img {{ width: 100%; height: 180px; object-fit: cover; background: #000; }}
  .card-body {{ padding: 12px; font-size: 13px; display: flex; flex-direction: column; gap: 6px; flex-grow: 1; }}
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; text-transform: uppercase; background: #3b82f6; color: white; width: fit-content; }}
  .preds {{ font-size: 11px; color: #94a3b8; background: #0f172a; padding: 6px; border-radius: 4px; }}
  .options {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-top: 6px; }}
  .options label {{ display: flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer; padding: 4px; border-radius: 4px; background: #334155; }}
  .options label:hover {{ background: #475569; }}
  .highlight {{ outline: 2px solid #f59e0b; background: rgba(245, 158, 11, 0.15) !important; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>HackBlox 2026 · Human Review Tool</h1>
    <div style="font-size: 13px; color: #94a3b8;">Reviewing {len(queue)} hard boundary cases. Select the true class for each image.</div>
  </div>
  <div style="display: flex; gap: 10px; align-items: center;">
    <span id="counter" style="font-weight: 600; color: #38bdf8;">0 / {len(queue)} reviewed</span>
    <button class="btn" onclick="exportCSV()">Export CSV</button>
  </div>
</div>

<div class="grid">
"""

for item in queue:
    img_rel = f"../data/train/undefined/{item['filename']}"
    t1 = item["top1_class"]
    t2 = item["top2_class"]

    html += f"""
  <div class="card" id="card-{item['id']}">
    <img src="{img_rel}" loading="lazy" alt="{item['filename']}" onclick="window.open('{img_rel}', '_blank')">
    <div class="card-body">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: bold;">#{item['order']} ({item['filename']})</span>
        <span class="badge">{item['pair_type']}</span>
      </div>
      <div class="preds">
        Seeds: S42: {CLASSES[item['pred_seed42']]} ({item['conf_seed42']:.2f}) | S43: {CLASSES[item['pred_seed43']]} ({item['conf_seed43']:.2f}) | S44: {CLASSES[item['pred_seed44']]} ({item['conf_seed44']:.2f})
      </div>
      <div class="options">
"""
    for c_idx, c_name in enumerate(CLASSES):
        hl = "highlight" if c_idx in [t1, t2] else ""
        checked = "checked" if c_idx == t1 else ""
        html += f"""        <label class="{hl}">
          <input type="radio" name="sample-{item['id']}" value="{c_idx}" {checked} onchange="updateCount()">
          [{c_idx}] {c_name}
        </label>
"""
    html += f"""      </div>
    </div>
  </div>
"""

html += f"""
</div>

<script>
const queue = {json.dumps(queue)};

function updateCount() {{
  let count = 0;
  queue.forEach(item => {{
    const checked = document.querySelector(`input[name="sample-${{item.id}}"]:checked`);
    if (checked) count++;
  }});
  document.getElementById('counter').innerText = `${{count}} / ${{queue.length}} reviewed`;
}}

function exportCSV() {{
  let csv = "id,image,label,label_name,pair_type,weight,source,reviewed_by,timestamp\\n";
  const classNames = ["buildings", "forest", "glacier", "mountain", "sea", "street"];
  const now = new Date().toISOString();

  queue.forEach(item => {{
    const checked = document.querySelector(`input[name="sample-${{item.id}}"]:checked`);
    if (checked) {{
      const lbl = parseInt(checked.value);
      csv += `${{item.id}},"${{item.image_path}}",${{lbl}},${{classNames[lbl]}},${{item.pair_type}},1.0,human_reviewed,expert_human,${{now}}\\n`;
    }}
  }});

  const blob = new Blob([csv], {{ type: 'text/csv' }});
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.setAttribute('href', url);
  a.setAttribute('download', 'human_labels.csv');
  a.click();
}}

updateCount();
</script>
</body>
</html>
"""

with open(html_out, "w") as f:
    f.write(html)
print(f"[OK] Generated HTML review tool: {html_out}")
