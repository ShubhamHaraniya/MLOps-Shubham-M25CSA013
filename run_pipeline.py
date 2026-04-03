"""
run_pipeline.py — Run the full Assignment 5 pipeline from the project root.
Works with or without Docker. Just run: python run_pipeline.py

Usage:
    python run_pipeline.py --q1             # Q1 only (all experiments)
    python run_pipeline.py --q2             # Q2 only
    python run_pipeline.py --q1 --q2        # both (default)
    python run_pipeline.py --q1 --exp 5     # Q1 single experiment
    python run_pipeline.py --optuna         # Optuna HPO after Q1
    python run_pipeline.py --push_hf        # push best model to HuggingFace
    python run_pipeline.py --device cpu     # force CPU (slow but functional)
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Q1   = ROOT / "q1_vit_lora"
Q2   = ROOT / "q2_adversarial"


def run(script_path, extra_args=None, cwd=None):
    """Run a Python script as a subprocess, inheriting stdout/stderr."""
    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"ERROR: {script_path.name} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Assignment 5 full pipeline runner")
    parser.add_argument("--q1",      action="store_true", help="Run Q1")
    parser.add_argument("--q2",      action="store_true", help="Run Q2")
    parser.add_argument("--optuna",  action="store_true",
                        help="Run Optuna HPO after Q1 experiments")
    parser.add_argument("--push_hf", action="store_true",
                        help="Push best Q1 model to HuggingFace after training")
    parser.add_argument("--exp",     type=int, default=-1,
                        help="Q1 experiment id. -1 = all (default)")
    parser.add_argument("--trials",  type=int, default=20,
                        help="Number of Optuna trials (default 20)")
    parser.add_argument("--device",  type=str, default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    # if neither flag given, run everything
    if not args.q1 and not args.q2:
        args.q1 = args.q2 = True

    device = ["--device", args.device]

    # ── Q1 ──────────────────────────────────────────────────────────────
    if args.q1:
        print("\n" + "█"*60)
        print("  Q1: ViT-S Fine-tuning on CIFAR-100")
        print("█"*60)

        # Train all / single experiment
        run(Q1 / "train.py", ["--exp", str(args.exp)] + device)

        # Test
        run(Q1 / "test.py", ["--exp", str(args.exp)] + device)

        # Optuna HPO
        if args.optuna:
            run(Q1 / "optuna_search.py",
                ["--trials", str(args.trials), "--retrain"] + device)

        # Push to HuggingFace
        if args.push_hf:
            run(Q1 / "push_to_hf.py")

    # ── Q2 ──────────────────────────────────────────────────────────────
    if args.q2:
        print("\n" + "█"*60)
        print("  Q2: Adversarial Attacks (IBM ART)")
        print("█"*60)

        # Step 1: Train ResNet-18 from scratch
        run(Q2 / "train_resnet18.py", device)

        # Step 2: FGSM from scratch
        run(Q2 / "fgsm_scratch.py", device)

        # Step 3: FGSM using ART
        run(Q2 / "fgsm_art.py", device)

        # Step 4: Train PGD + BIM detectors
        run(Q2 / "train_detector.py", ["--attack", "both"] + device)

        # Step 5: Evaluate detectors
        run(Q2 / "evaluate_detector.py", ["--attack", "both"] + device)

        # Step 6: Generate all visualisations and log to WandB
        run(Q2 / "visualize.py", device)

    print("\n" + "✓"*60)
    print("  Pipeline complete! Check results/ and weights/ directories.")
    print("  Add your WandB and HuggingFace links to README.md")
    print("✓"*60 + "\n")


if __name__ == "__main__":
    main()
