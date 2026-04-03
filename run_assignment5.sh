#!/bin/bash
#SBATCH --job-name=assignment5
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn05
#SBATCH --output=logs/assignment5_%j.out
#SBATCH --error=logs/assignment5_%j.err
#SBATCH --time=12:00:00

# ============================================================
# Assignment 5 — Full Pipeline Runner
# Submit with:  sbatch run_assignment5.sh
# Or run on an interactive node:
#   srun --pty --partition=mtech --gres=gpu:1 --exclude=cn05 bash
#   bash run_assignment5.sh
# ============================================================

set -e  # exit on first error

# 1. Load Python module
echo "==> Loading Python module..."
module load python/3.10.pytorch

# Absolute path to the project — SLURM moves the script so dirname $0 won't work
ROOT="/csehome/m25csa013/Assignment 5"

# 2. Load credentials from .env
echo "==> Loading credentials..."
source "$ROOT/.env"
export WANDB_API_KEY HF_TOKEN HF_REPO_ID WANDB_PROJECT

# 3. Make sure user packages are on PATH
export PATH="$HOME/.local/bin:$PATH"

# 4. Install any missing dependencies
echo "==> Installing dependencies..."
pip3 install --quiet --user \
    timm peft wandb optuna huggingface_hub \
    "adversarial-robustness-toolbox[pytorch]" \
    safetensors scikit-learn tqdm

# create log dir
mkdir -p "$ROOT/logs"

# ──────────────────────────────────────────────
# Q1 — ViT-S fine-tuning on CIFAR-100 with LoRA
# ──────────────────────────────────────────────
# echo ""
# echo "============================================"
# echo " Q1: ViT-S + LoRA on CIFAR-100"
# echo "============================================"
# cd "$ROOT/q1_vit_lora"

# # Train all 10 experiments (exp 0 = no-LoRA baseline; exp 1-9 = LoRA grid)
# python3 train.py --exp -1 --device cuda

# # Test all checkpoints
# python3 test.py --exp -1 --device cuda

# # Optuna hyperparameter search (20 trials) + retrain best config
# python3 optuna_search.py --trials 20 --retrain --device cuda

# # Push best model to HuggingFace Hub
# python3 push_to_hf.py

# ──────────────────────────────────────────────
# Q2 — Adversarial Attacks with IBM ART
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo " Q2: Adversarial Attacks (FGSM / PGD / BIM)"
echo "============================================"
cd "$ROOT/q2_adversarial"

# Step 1: Train ResNet-18 from scratch (target >= 72%)
python3 train_resnet18.py --epochs 30 --device cuda

# Step 2: FGSM from scratch (no ART)
python3 fgsm_scratch.py --device cuda

# Step 3: FGSM using IBM ART + visual comparison
python3 fgsm_art.py --device cuda

# Step 4: Train adversarial detectors for PGD + BIM
python3 train_detector.py --attack both --device cuda --n_train 10000

# Step 5: Evaluate detectors (confusion matrix, AUC)
python3 evaluate_detector.py --attack both --device cuda

# Step 6: Generate all visualisations + log samples to WandB
python3 visualize.py --device cuda

echo ""
echo "============================================"
echo " All done! Check WandB project: $WANDB_PROJECT"
echo " HuggingFace model: https://huggingface.co/$HF_REPO_ID"
echo "============================================"
