"""
Re-evaluate the model from HuggingFace Hub.

Downloads the model from the HF repo and runs the same evaluation pipeline,
allowing comparison with the local model evaluation results.

Usage:
    python evaluate_from_hub.py --repo_name spidey1807/mlops_ass3
"""

import os
import json
import argparse
import shutil
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from src.data import load_all_genres, split_data
from src.utils import create_label_mappings, encode_data, compute_metrics


RESULTS_DIR = './results'
PLOTS_DIR = './results/plots'
MAX_LENGTH = 256


def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Percentage (%)')

    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix - Hub Model (Normalized %)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    thresh = cm_pct.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_pct[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")


def plot_per_class_metrics(true_labels, pred_labels, class_names, save_path):
    """Generate and save a grouped bar chart of per-class precision, recall, and F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=class_names, average=None, zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#4C72B0')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#55A868')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#C44E52')

    ax.set_xlabel('Genre')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics - Hub Model')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-class metrics chart saved to: {save_path}")


def plot_overall_metrics(eval_results, save_path):
    """Generate and save a bar chart of overall evaluation metrics."""
    metrics = {
        'Accuracy': eval_results['eval_accuracy'],
        'Precision': eval_results['eval_precision'],
        'Recall': eval_results['eval_recall'],
        'F1-Score': eval_results['eval_f1'],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, width=0.5, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics - Hub Model')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Overall metrics chart saved to: {save_path}")


def plot_comparison(local_results, hub_results, save_path):
    """Generate a side-by-side comparison chart of local vs hub model."""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']

    local_vals = [local_results.get(k, 0) for k in metric_keys]
    hub_vals = [hub_results.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, local_vals, width, label='Local Model', color='#4C72B0')
    bars2 = ax.bar(x + width/2, hub_vals, width, label='Hub Model', color='#C44E52')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Local vs Hub Model Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison chart saved to: {save_path}")


def evaluate_from_hub(repo_name, token=None, device=None):
    """
    Download model from HuggingFace Hub and evaluate on test data.

    Args:
        repo_name: HuggingFace repository ID (e.g., 'username/model-name').
        token: HuggingFace API token (optional for public repos).
        device: Device to use. Auto-detected if None.

    Returns:
        Dictionary of evaluation metrics.
    """
    os.environ["WANDB_DISABLED"] = "true"

    if token is None:
        token = os.environ.get('HF_TOKEN')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Loading test data...")
    print("="*60)
    genre_reviews = load_all_genres(head=10000, sample_size=2000)
    _, _, test_texts, test_labels = split_data(genre_reviews)

    # ── Step 2: Download model from Hub ────────────────────────
    print("\n" + "="*60)
    print(f"STEP 2: Downloading model from HuggingFace Hub: {repo_name}")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(repo_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(repo_name, token=token).to(device)

    # ── Step 3: Prepare data ───────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Encoding test data...")
    print("="*60)

    # Try to download label mappings from the repo
    try:
        from huggingface_hub import hf_hub_download
        mappings_path = hf_hub_download(repo_id=repo_name, filename='label_mappings.json', token=token)
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
            label2id = mappings['label2id']
            id2label = {int(k): v for k, v in mappings['id2label'].items()}
    except Exception:
        print("Could not load label mappings from Hub, creating from test labels...")
        label2id, id2label = create_label_mappings(test_labels)

    test_dataset = encode_data(tokenizer, test_texts, test_labels, label2id, MAX_LENGTH)

    # ── Step 4: Run evaluation ─────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Running evaluation (Hub model)...")
    print("="*60)

    hub_eval_output_dir = './hub_eval_output'
    training_args = TrainingArguments(
        output_dir=hub_eval_output_dir,
        per_device_eval_batch_size=16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate(test_dataset)
    print(f"\nHub Model Evaluation Results:")
    print(f"  Loss:      {eval_results['eval_loss']:.4f}")
    print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")
    print(f"  F1:        {eval_results['eval_f1']:.4f}")

    # Classification report
    print("\n" + "="*60)
    print("STEP 5: Detailed Classification Report (Hub Model)")
    print("="*60)
    predicted_results = trainer.predict(test_dataset)
    predicted_labels = predicted_results.predictions.argmax(-1)
    predicted_labels = [id2label[l] for l in predicted_labels.flatten().tolist()]
    report = classification_report(test_labels, predicted_labels)
    print(report)

    # ── Step 6: Generate visualizations ────────────────────────
    print("\n" + "="*60)
    print("STEP 6: Generating evaluation plots...")
    print("="*60)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    class_names = sorted(set(test_labels))

    plot_confusion_matrix(
        test_labels, predicted_labels, class_names,
        os.path.join(PLOTS_DIR, 'hub_confusion_matrix.png')
    )
    plot_per_class_metrics(
        test_labels, predicted_labels, class_names,
        os.path.join(PLOTS_DIR, 'hub_per_class_metrics.png')
    )
    plot_overall_metrics(
        eval_results,
        os.path.join(PLOTS_DIR, 'hub_overall_metrics.png')
    )

    # ── Step 7: Save and compare ───────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, 'hub_eval_results.json')

    hub_results = {
        'source': 'huggingface_hub',
        'model_path': repo_name,
        'eval_loss': eval_results['eval_loss'],
        'eval_accuracy': eval_results['eval_accuracy'],
        'eval_precision': eval_results['eval_precision'],
        'eval_recall': eval_results['eval_recall'],
        'eval_f1': eval_results['eval_f1'],
        'classification_report': report,
    }

    with open(results_path, 'w') as f:
        json.dump(hub_results, f, indent=2)
    print(f"\nHub results saved to: {results_path}")

    # Compare with local results if available
    local_results_path = os.path.join(RESULTS_DIR, 'eval_results.json')
    if os.path.exists(local_results_path):
        with open(local_results_path, 'r') as f:
            local_results = json.load(f)

        print("\n" + "="*60)
        print("COMPARISON: Local vs. Hub Model")
        print("="*60)
        print(f"{'Metric':<15} {'Local':>10} {'Hub':>10} {'Diff':>10}")
        print("-" * 50)
        for metric in ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1', 'eval_loss']:
            local_val = local_results.get(metric, 0)
            hub_val = hub_results.get(metric, 0)
            diff = hub_val - local_val
            sign = '+' if diff >= 0 else ''
            print(f"{metric:<15} {local_val:>10.4f} {hub_val:>10.4f} {sign}{diff:>9.4f}")

        # Generate comparison chart
        plot_comparison(
            local_results, hub_results,
            os.path.join(PLOTS_DIR, 'local_vs_hub_comparison.png')
        )
    else:
        print("\nNo local evaluation results found for comparison.")

    # Clean up Trainer temp directory
    if os.path.exists(hub_eval_output_dir):
        shutil.rmtree(hub_eval_output_dir, ignore_errors=True)

    return hub_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model from HuggingFace Hub")
    parser.add_argument('--repo_name', type=str, required=True,
                        help='HuggingFace repo name (e.g., username/model-name)')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token (or set HF_TOKEN env var)')
    args = parser.parse_args()

    evaluate_from_hub(repo_name=args.repo_name, token=args.token)
