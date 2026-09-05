#!/bin/bash
set -e

echo "=== Starting Seed 43 Training on train_0006b ==="
.venv/bin/python src/train.py --table-url /home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0006b --epochs 15 --lr 0.0001 --advanced --seed 43 --save-name scratch/clean_model_seed43_v6b.pth --skip-metrics

echo "=== Starting Seed 44 Training on train_0006b ==="
.venv/bin/python src/train.py --table-url /home/h3r0-k1ll3r/.local/share/3LC/projects/Intel-Scene/datasets/intel-scene/tables/train_0006b --epochs 15 --lr 0.0001 --advanced --seed 44 --save-name scratch/clean_model_seed44_v6b.pth --skip-metrics

echo "=== Evaluating All Ensembles (train_0005, train_0006, train_0006b) ==="
.venv/bin/python evaluate_and_compare_v6b.py

echo "=== Generating Predictions for train_0006b Ensemble ==="
.venv/bin/python src/ensemble.py --mode seeds --checkpoints scratch/clean_model_seed42_v6b.pth scratch/clean_model_seed43_v6b.pth scratch/clean_model_seed44_v6b.pth --output submission_train0006b.csv

echo "=== All Steps Finished Successfully ==="
