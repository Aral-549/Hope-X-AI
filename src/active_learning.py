"""
Active Learning and 3LC Table Revision Engine for Intel Scene Classification.
Strictly follows HackBlox 2026 / 3LC Challenge rules:
- Max 3,000 active samples (weight=1.0)
- Full 3LC table lineage with input_tables and versioned revisions
- Reproducible, seeded selection
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tlc
from src.utils import load_config, set_seed

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
ALL_CLASSES = CLASSES + ["undefined"]

SCHEMAS = {
    "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
    "image": tlc.ImagePath,
    "label": tlc.CategoricalLabel("label", classes=ALL_CLASSES),
    "weight": tlc.SampleWeightSchema(),
}


def get_latest_run_parquet(project_name="Intel-Scene"):
    """Find the most recent run's reduced_0000.parquet in the 3LC project directory."""
    runs_dir = Path.home() / ".local/share/3LC/projects" / project_name / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found at {runs_dir}")
    
    runs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.endswith(".lock")]
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for r in runs:
        parquet_file = r / "reduced_0000" / "reduced_0000.parquet"
        if parquet_file.exists():
            return parquet_file, r.name
            
    raise FileNotFoundError(f"No reduced_0000.parquet found in any run under {runs_dir}")


def load_pool_dataframe(train_table, parquet_path):
    """
    Merge 3LC train table rows with the latest run's predictions, confidence, and UMAP embeddings.
    """
    df_metrics = pd.read_parquet(parquet_path)
    
    rows = []
    # Use table_rows for weight and alias, and table[i] for resolved file path
    for i, row in enumerate(train_table.table_rows):
        row_dict = dict(row)
        resolved_img_path = train_table[i]["image"]
        rows.append({
            "id": row_dict["id"],
            "image": resolved_img_path,
            "image_alias": row_dict["image"],
            "current_label": row_dict["label"],
            "current_weight": float(row_dict.get("weight", 0.0)),
        })
    df_table = pd.DataFrame(rows)
    
    merged = pd.merge(df_table, df_metrics, left_on="id", right_on="example_id", how="left")
    
    # Unpack UMAP 3D embeddings
    if "embeddings_mean_67_umap" in merged.columns:
        umap_coords = np.vstack(merged["embeddings_mean_67_umap"].values)
        merged["umap_x"] = umap_coords[:, 0]
        merged["umap_y"] = umap_coords[:, 1]
        merged["umap_z"] = umap_coords[:, 2]
    
    return merged


def select_loop1_uncertainty_diversity(df, n_to_add=800, seed=42):
    """
    Loop 1 Selection Strategy:
    Uncertainty / Margin sampling with embedding diversity across the undefined pool.
    - Picks low confidence (high uncertainty) samples near decision boundaries
    - Stratifies across predicted classes for balanced representation
    - Uses k-means on 3D UMAP coordinates within each class to ensure spatial diversity
    """
    np.random.seed(seed)
    
    candidates = df[df["current_weight"] == 0.0].copy()
    print(f"Loop 1 Candidate pool: {len(candidates)} unselected samples")
    
    candidates["uncertainty"] = 1.0 - candidates["confidence"]
    
    samples_per_class = n_to_add // len(CLASSES)  # 133
    remainder = n_to_add % len(CLASSES)           # 2
    
    selected_ids = []
    for c_idx in range(len(CLASSES)):
        quota = samples_per_class + (1 if c_idx < remainder else 0)
        c_candidates = candidates[candidates["predicted"] == c_idx].copy()
        
        if len(c_candidates) <= quota:
            selected_ids.extend(c_candidates["id"].tolist())
            continue
            
        n_clusters = min(8, len(c_candidates))
        umap_feats = c_candidates[["umap_x", "umap_y", "umap_z"]].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(umap_feats)
        c_candidates["cluster"] = kmeans.labels_
        
        per_cluster = quota // n_clusters
        cluster_rem = quota % n_clusters
        class_selected = []
        for cluster_id in range(n_clusters):
            cluster_subset = c_candidates[c_candidates["cluster"] == cluster_id]
            take_count = per_cluster + (1 if cluster_id < cluster_rem else 0)
            top_uncertain = cluster_subset.sort_values(by="uncertainty", ascending=False).head(take_count)
            class_selected.extend(top_uncertain["id"].tolist())
            
        if len(class_selected) < quota:
            remaining = c_candidates[~c_candidates["id"].isin(class_selected)]
            needed = quota - len(class_selected)
            fillers = remaining.sort_values(by="uncertainty", ascending=False).head(needed)
            class_selected.extend(fillers["id"].tolist())
            
        selected_ids.extend(class_selected)
        print(f"  Class {c_idx} ({CLASSES[c_idx]}): Selected {len(class_selected)} / quota {quota}")
        
    print(f"Total Loop 1 selected samples: {len(selected_ids)}")
    return selected_ids


def select_loop2_error_focused(df, n_to_add=800, seed=42):
    """
    Loop 2 Selection Strategy:
    Class-balanced error-focused sampling targeting systematically weaker classes from Loop 1.
    Heavily targets glacier (2), mountain (3), buildings (0), and street (5).
    """
    np.random.seed(seed)
    candidates = df[df["current_weight"] == 0.0].copy()
    print(f"Loop 2 Candidate pool: {len(candidates)} unselected samples")
    
    candidates["uncertainty"] = 1.0 - candidates["confidence"]
    
    quotas = {0: 160, 1: 70, 2: 175, 3: 175, 4: 70, 5: 150}
    
    selected_ids = []
    for c_idx, quota in quotas.items():
        c_candidates = candidates[candidates["predicted"] == c_idx].copy()
        if len(c_candidates) <= quota:
            selected_ids.extend(c_candidates["id"].tolist())
            continue
            
        n_clusters = min(6, len(c_candidates))
        umap_feats = c_candidates[["umap_x", "umap_y", "umap_z"]].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(umap_feats)
        c_candidates["cluster"] = kmeans.labels_
        
        per_cluster = quota // n_clusters
        cluster_rem = quota % n_clusters
        class_selected = []
        for cluster_id in range(n_clusters):
            cluster_subset = c_candidates[c_candidates["cluster"] == cluster_id]
            take_count = per_cluster + (1 if cluster_id < cluster_rem else 0)
            top_uncertain = cluster_subset.sort_values(by="uncertainty", ascending=False).head(take_count)
            class_selected.extend(top_uncertain["id"].tolist())
            
        if len(class_selected) < quota:
            remaining = c_candidates[~c_candidates["id"].isin(class_selected)]
            needed = quota - len(class_selected)
            fillers = remaining.sort_values(by="uncertainty", ascending=False).head(needed)
            class_selected.extend(fillers["id"].tolist())
            
        selected_ids.extend(class_selected)
        print(f"  Class {c_idx} ({CLASSES[c_idx]}): Selected {len(class_selected)} / quota {quota}")
        
    print(f"Total Loop 2 selected samples: {len(selected_ids)}")
    return selected_ids


def select_loop3_hard_negatives(df, n_to_add=600, seed=42):
    """
    Loop 3 Selection Strategy:
    Strict Hard-Negative Mining Pass.
    Spends remaining budget strictly on known confused pairs:
    1. Glacier (2) vs Mountain (3) vs Sea (4)
    2. Buildings (0) vs Street (5)
    """
    np.random.seed(seed)
    candidates = df[df["current_weight"] == 0.0].copy()
    print(f"Loop 3 Candidate pool: {len(candidates)} unselected samples")
    
    candidates["boundary_ambiguity"] = -np.abs(candidates["confidence"] - 0.5)
    
    quotas = {2: 120, 3: 120, 4: 120, 0: 120, 5: 120}
    
    selected_ids = []
    for c_idx, quota in quotas.items():
        c_candidates = candidates[candidates["predicted"] == c_idx].copy()
        if len(c_candidates) <= quota:
            selected_ids.extend(c_candidates["id"].tolist())
            continue
            
        top_boundary = c_candidates.sort_values(by="boundary_ambiguity", ascending=False).head(quota)
        selected_ids.extend(top_boundary["id"].tolist())
        print(f"  Hard boundary Class {c_idx} ({CLASSES[c_idx]}): Selected {len(top_boundary)} / quota {quota}")
        
    print(f"Total Loop 3 hard-negative samples: {len(selected_ids)}")
    return selected_ids


def annotate_samples(df_selected, device="cuda"):
    """
    High-fidelity multi-prompt zero-shot scene annotator.
    Mimics expert human labeling in the 3LC Dashboard.
    """
    import open_clip
    
    print(f"\n[ANNOTATOR] Annotating {len(df_selected)} selected samples using multi-prompt vision model...")
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    prompt_templates = [
        "a photo of {c}",
        "a scenic photograph of a {c}",
        "an outdoor landscape photo of {c}",
        "a clear view of {c}",
    ]
    
    class_descs = {
        0: "buildings, houses, and architectural skyline",
        1: "forest with lush green trees and foliage",
        2: "glacier with ice, snow, and polar icebergs",
        3: "rocky mountain peaks and slopes",
        4: "sea, ocean water, waves, and coast",
        5: "street with urban roads, asphalt, and pavement",
    }
    
    text_embeddings_list = []
    with torch.no_grad():
        for c_idx in range(6):
            c_desc = class_descs[c_idx]
            prompts = [tpl.format(c=c_desc) for tpl in prompt_templates] + [f"a photo of {CLASSES[c_idx]}"]
            tokens = tokenizer(prompts).to(device)
            embeddings = clip_model.encode_text(tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            class_mean = embeddings.mean(dim=0, keepdim=True)
            class_mean /= class_mean.norm(dim=-1, keepdim=True)
            text_embeddings_list.append(class_mean)
            
        all_text_features = torch.cat(text_embeddings_list, dim=0)
        
    annotations = {}
    from torch.utils.data import Dataset, DataLoader

    class ImageListDataset(Dataset):
        def __init__(self, df, transform):
            self.df = df
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row["image"]).convert("RGB")
            return self.transform(img), row["id"]

    ds = ImageListDataset(df_selected, preprocess)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    with torch.no_grad():
        for batch_imgs, batch_ids in loader:
            batch_imgs = batch_imgs.to(device)
            img_feats = clip_model.encode_image(batch_imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            similarities = (100.0 * img_feats @ all_text_features.T).softmax(dim=-1)
            pred_classes = similarities.argmax(dim=-1).cpu().numpy()
            confs = similarities.gather(1, torch.tensor(pred_classes, device=device).unsqueeze(1)).squeeze(1).cpu().numpy()
            
            for b_id, pred_cls, conf in zip(batch_ids.numpy(), pred_classes, confs):
                annotations[int(b_id)] = {"label": int(pred_cls), "annotator_conf": float(conf)}

    print(f"[ANNOTATOR] Completed annotations for {len(annotations)} samples.")
    return annotations


def create_revision(train_table, new_annotations, loop_idx, description=None):
    """
    Create a new versioned 3LC table revision with explicit input_tables lineage.
    Enforces hard rule: weight=1 count <= 3000.
    """
    config = load_config()
    max_weight1 = config["data"]["max_weight1_rows"]
    
    desc = description or f"Intel Scene train revision - Loop {loop_idx}"
    print(f"\n[REVISION] Creating 3LC Table Revision for Loop {loop_idx}...")
    print(f"  Parent table URL: {train_table.url}")
    
    writer = tlc.TableWriter(
        table_name="train",
        dataset_name=config["project"]["dataset_name"],
        project_name=config["project"]["name"],
        description=desc,
        column_schemas=SCHEMAS,
        input_tables=[train_table.url],
        if_exists="rename",
    )
    
    active_count = 0
    updated_count = 0
    
    for row in train_table.table_rows:
        row_dict = dict(row)
        row_id = row_dict["id"]
        
        if row_id in new_annotations:
            ann = new_annotations[row_id]
            label = ann["label"]
            weight = 1.0
            updated_count += 1
            active_count += 1
        else:
            label = row_dict["label"]
            weight = float(row_dict.get("weight", 0.0))
            if weight > 0:
                active_count += 1
                
        writer.add_row({
            "id": row_id,
            "image": row_dict["image"],
            "label": label,
            "weight": weight,
        })
        
    print(f"  Newly activated rows: {updated_count} | Total active rows: {active_count} / {max_weight1}")
    if active_count > max_weight1:
        raise ValueError(f"HARD RULE VIOLATION: Active samples {active_count} exceeds limit {max_weight1}!")
        
    new_table = writer.finalize()
    print(f"[OK] Created revision: {new_table.url}")
    
    meta = {
        "loop": loop_idx,
        "table_url": str(new_table.url),
        "parent_table_url": str(train_table.url),
        "total_rows": len(new_table),
        "active_weight1_rows": active_count,
        "newly_annotated": updated_count,
        "description": desc,
    }
    meta_path = PROJECT_ROOT / "3lc_tables" / f"train_loop{loop_idx}_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"[METADATA] Saved revision metadata to {meta_path}")
    return new_table, active_count
