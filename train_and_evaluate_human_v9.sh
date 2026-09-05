#!/bin/bash
set -e

TABLE_URL="/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0009"

echo "======================================================================"
echo "  Starting Training on Authentic Human-Reviewed Table train_0009"
echo "  Total Active Rows: Exactly 3,000 / 3,000"
echo "======================================================================"

echo -e "\n--- Training Seed 42 ---"
.venv/bin/python src/train.py --table-url "$TABLE_URL" --epochs 15 --lr 0.0001 --advanced --seed 42 --save-name scratch/clean_model_seed42_v9.pth --skip-metrics

echo -e "\n--- Training Seed 43 ---"
.venv/bin/python src/train.py --table-url "$TABLE_URL" --epochs 15 --lr 0.0001 --advanced --seed 43 --save-name scratch/clean_model_seed43_v9.pth --skip-metrics

echo -e "\n--- Training Seed 44 ---"
.venv/bin/python src/train.py --table-url "$TABLE_URL" --epochs 15 --lr 0.0001 --advanced --seed 44 --save-name scratch/clean_model_seed44_v9.pth --skip-metrics

echo -e "\n--- Running Comparative Evaluation ---"
.venv/bin/python evaluate_and_compare_v9.py

echo -e "\n--- Generating Test Submission for train_0009 Ensemble ---"
.venv/bin/python src/ensemble.py --mode seeds --checkpoints scratch/clean_model_seed42_v9.pth scratch/clean_model_seed43_v9.pth scratch/clean_model_seed44_v9.pth --output submission_train0009.csv

echo -e "\n--- Updating Primary submission.csv ---"
cp submission_train0009.csv submission.csv

echo -e "\n======================================================================"
echo "  [ALL DONE] Authentic Human train_0009 Ensemble Complete!"
echo "======================================================================"
