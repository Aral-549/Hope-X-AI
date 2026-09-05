#!/bin/bash
set -e

TABLE_URL="/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0009"

echo "======================================================================"
echo "  Step 1: Training Class-Balanced Model on train_0009 (Seed 42)"
echo "======================================================================"

.venv/bin/python src/train.py --table-url "$TABLE_URL" --epochs 15 --lr 0.0001 --advanced --balanced-sampler --seed 42 --save-name scratch/clean_model_seed42_v9_balanced.pth --skip-metrics

echo -e "\n======================================================================"
echo "  Evaluating Balanced Model & Stacking Blends"
echo "======================================================================"
.venv/bin/python eval_balanced_v9.py
