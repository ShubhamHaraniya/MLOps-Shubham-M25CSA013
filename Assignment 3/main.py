"""
Main entry point for the Goodreads Genre Classification pipeline.

Usage:
    python main.py --train                    # Train the model
    python main.py --eval                     # Evaluate local model
    python main.py --push --repo_name <repo>  # Push to HuggingFace
    python main.py --all --repo_name <repo>   # Run everything
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Goodreads Genre Classification with DistilBERT"
    )
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the local model')
    parser.add_argument('--push', action='store_true', help='Push model to HuggingFace Hub')
    parser.add_argument('--eval_hub', action='store_true', help='Evaluate model from HuggingFace Hub')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--repo_name', type=str, default=None,
                        help='HuggingFace repo name (required for --push and --eval_hub)')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace token (or set HF_TOKEN env var)')
    parser.add_argument('--model_dir', type=str, default='./saved_model',
                        help='Directory for saving/loading the model')
    args = parser.parse_args()

    # Disable wandb
    os.environ["WANDB_DISABLED"] = "true"

    if args.all:
        args.train = True
        args.eval = True
        args.push = True
        args.eval_hub = True

    if not any([args.train, args.eval, args.push, args.eval_hub]):
        parser.print_help()
        return

    # ── Training ───────────────────────────────────────────────
    if args.train:
        print("\n" + "#"*60)
        print("# TRAINING PIPELINE")
        print("#"*60)
        from src.train import train
        train(output_dir=args.model_dir)

    # ── Local Evaluation ───────────────────────────────────────
    if args.eval:
        print("\n" + "#"*60)
        print("# LOCAL EVALUATION")
        print("#"*60)
        from src.eval import evaluate
        evaluate(model_path=args.model_dir)

    # ── Push to Hub ────────────────────────────────────────────
    if args.push:
        if not args.repo_name:
            print("ERROR: --repo_name is required for pushing to HuggingFace Hub")
            return
        print("\n" + "#"*60)
        print("# PUSHING TO HUGGINGFACE HUB")
        print("#"*60)
        from src.push_to_hub import push_to_hub
        push_to_hub(model_dir=args.model_dir, repo_name=args.repo_name, token=args.token)

    # ── Hub Evaluation ─────────────────────────────────────────
    if args.eval_hub:
        if not args.repo_name:
            print("ERROR: --repo_name is required for evaluating from HuggingFace Hub")
            return
        print("\n" + "#"*60)
        print("# HUB MODEL EVALUATION")
        print("#"*60)
        from evaluate_from_hub import evaluate_from_hub
        evaluate_from_hub(repo_name=args.repo_name, token=args.token)

    print("\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)


if __name__ == "__main__":
    main()
