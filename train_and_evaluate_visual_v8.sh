#!/bin/bash
set -e

TABLE_URL="/home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0008"

echo "======================================================================"
echo "  Starting Training on Visually Verified Table train_0008 (2,971 rows)"
echo "======================================================================"

echo -e "\n--- Training Seed 42 ---"
.venv/bin/python src/train.py --table-url "$TABLE_URL" --epochs 15 --lr 0.0001 --advanced --seed 42 --save-name scratch/clean_model_seed42_v8.pth --skip-metrics

echo -e "\n--- Training Seed 43 ---"
.venv/bin/python src/train.py --table-url "$TABLE_URL" --epochs 15 --lr 0.0001 --advanced --seed 43 --save-name scratch/clean_model_seed43_v8.pth --skip-metrics

echo -e "\n--- Training Seed 44 ---"
.venv/bin/python src/train.py --table-url "$TABLE_URL" --epochs 15 --lr 0.0001 --advanced --seed 44 --save-name scratch/clean_model_seed44_v8.pth --skip-metrics

echo -e "\n--- Running Comparative Evaluation ---"
.venv/bin/python evaluate_and_compare_v8.py

echo -e "\n--- Generating Test Submission for train_0008 Ensemble ---"
.venv/bin/python src/ensemble.py --mode seeds --checkpoints scratch/clean_model_seed42_v8.pth scratch/clean_model_seed43_v8.pth scratch/clean_model_seed44_v8.pth --output submission_train0008.csv

echo -e "\n======================================================================"
echo "  [ALL DONE] train_0008 Training and Evaluation Complete!"
echo "======================================================================"
