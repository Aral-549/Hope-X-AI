import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.metrics import recall_score
import tlc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ResNet18Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

val_table = tlc.Table.from_names(project_name="Intel-Scene", dataset_name="intel-scene", table_name="val").latest()
raw_samples = [(val_table[i]["image"], val_table[i]["label"]) for i in range(len(val_table))]
targets = np.array([s[1] for s in raw_samples])

# Load SWA model
model = ResNet18Classifier(num_classes=6, dropout_rate=0.3).to(device)
model.load_state_dict(torch.load("scratch/clean_model_seed42_v9_swa.pth", map_location=device))
model.eval()

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_std = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), norm])
tf_zoom = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(150), transforms.ToTensor(), norm])

angles = [-12, -6, 0, 6, 12]

probs_multicrop = []
probs_rotation = []

print("Running Rotation TTA and Multi-Crop TTA on SWA model...")
with torch.no_grad():
    for idx, (img_path, _) in enumerate(raw_samples):
        clean_path = str(img_path).replace("<INTEL_SCENE_DATA>/", "").replace("/home/h3r0-k1ll3r/Downloads/hackBlox/Round02/<INTEL_SCENE_DATA>/", str(PROJECT_ROOT) + "/")
        if not Path(clean_path).exists():
            clean_path = str(PROJECT_ROOT / clean_path)
        img = Image.open(clean_path).convert("RGB")
        
        # 1. 2-Scale Multi-Crop
        t_std = tf_std(img).unsqueeze(0).to(device)
        t_zoom = tf_zoom(img).unsqueeze(0).to(device)
        p_mc = 0.5 * (F.softmax(model(t_std), dim=1) + F.softmax(model(t_zoom), dim=1)).cpu().numpy()
        probs_multicrop.append(p_mc)
        
        # 2. Rotation TTA
        rot_sum = None
        for angle in angles:
            img_rot = transforms.functional.rotate(img, angle)
            t_rot = tf_std(img_rot).unsqueeze(0).to(device)
            p_rot = F.softmax(model(t_rot), dim=1)
            rot_sum = p_rot if rot_sum is None else rot_sum + p_rot
        p_rot_avg = (rot_sum / len(angles)).cpu().numpy()
        probs_rotation.append(p_rot_avg)

p_mc = np.concatenate(probs_multicrop, axis=0)
p_rot = np.concatenate(probs_rotation, axis=0)

acc_mc = 100.0 * (p_mc.argmax(1) == targets).mean()
acc_rot = 100.0 * (p_rot.argmax(1) == targets).mean()

# Blend: 50/50
p_blend = 0.5 * p_mc + 0.5 * p_rot
acc_blend = 100.0 * (p_blend.argmax(1) == targets).mean()

# Weighted blend search (0.7 mc + 0.3 rot, etc.)
p_blend_73 = 0.7 * p_mc + 0.3 * p_rot
acc_blend_73 = 100.0 * (p_blend_73.argmax(1) == targets).mean()

print("=" * 70)
print(f"ROTATION TTA EVALUATION ON SWA MODEL:")
print("=" * 70)
print(f"  2-Scale Multi-Crop Only            : {acc_mc:.2f}%")
print(f"  Rotation TTA Only (angles {angles}): {acc_rot:.2f}%")
print(f"  Combined (0.5 Multi-Crop + 0.5 Rot): {acc_blend:.2f}%")
print(f"  Combined (0.7 Multi-Crop + 0.3 Rot): {acc_blend_73:.2f}%")
print("=" * 70)
