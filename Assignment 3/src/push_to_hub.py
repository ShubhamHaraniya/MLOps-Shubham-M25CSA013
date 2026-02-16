"""
Push the fine-tuned model and tokenizer to Hugging Face Hub.

Usage:
    python -m src.push_to_hub --repo_name spidey1807/mlops_ass3

Requires HF_TOKEN environment variable with write permissions.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi


def push_to_hub(model_dir='./saved_model', repo_name=None, token=None):
    """
    Push the saved model and tokenizer to Hugging Face Hub.

    Args:
        model_dir: Local directory containing the saved model.
        repo_name: HuggingFace repository name (e.g., 'username/model-name').
        token: HuggingFace API token with write permissions.
    """
    if token is None:
        token = os.environ.get('HF_TOKEN')
    if token is None:
        raise ValueError("HF_TOKEN environment variable not set. Please provide a HuggingFace token.")

    if repo_name is None:
        raise ValueError("Please provide a repository name (e.g., 'username/model-name').")

    print(f"Loading model from: {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Pushing model to HuggingFace Hub: {repo_name}")
    model.push_to_hub(repo_name, token=token)
    tokenizer.push_to_hub(repo_name, token=token)

    # Also push the label mappings file if it exists
    label_mappings_path = os.path.join(model_dir, 'label_mappings.json')
    if os.path.exists(label_mappings_path):
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=label_mappings_path,
            path_in_repo='label_mappings.json',
            repo_id=repo_name,
        )
        print("Label mappings uploaded.")

    print(f"\nModel successfully pushed to: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument('--model_dir', type=str, default='./saved_model',
                        help='Local directory with saved model')
    parser.add_argument('--repo_name', type=str, required=True,
                        help='HuggingFace repo name (e.g., username/model-name)')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token (or set HF_TOKEN env var)')
    args = parser.parse_args()

    push_to_hub(
        model_dir=args.model_dir,
        repo_name=args.repo_name,
        token=args.token,
    )
