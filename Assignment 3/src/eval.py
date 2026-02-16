"""
Evaluation module for the fine-tuned BERT genre classifier.

Loads a model from a local directory, runs evaluation on test data,
generates visualizations (confusion matrix, per-class metrics), and saves results.
"""

import os
import json
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
           title='Confusion Matrix (Normalized %)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    # Annotate each cell with count and percentage
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
    ax.set_title('Per-Class Precision, Recall & F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on top of bars
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
    ax.set_title('Overall Evaluation Metrics')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Overall metrics chart saved to: {save_path}")


def evaluate(
    model_path='./saved_model',
    results_filename='eval_results.json',
    device=None,
    source_label='local',
):
    """
    Evaluate a fine-tuned model on Goodreads test data with visualizations.

    Args:
        model_path: Path to local model directory or HuggingFace repo ID.
        results_filename: Name of the results JSON file to save.
        device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        source_label: Label for the evaluation source ('local' or 'hub').

    Returns:
        Dictionary of evaluation metrics.
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Loading test data...")
    print("="*60)
    genre_reviews = load_all_genres(head=10000, sample_size=2000)
    _, _, test_texts, test_labels = split_data(genre_reviews)

    # ── Step 2: Load model and tokenizer ───────────────────────
    print("\n" + "="*60)
    print(f"STEP 2: Loading model from: {model_path}")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # ── Step 3: Prepare data ───────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Encoding test data...")
    print("="*60)

    # Load label mappings
    label_mappings_path = os.path.join(model_path, 'label_mappings.json')
    if os.path.exists(label_mappings_path):
        with open(label_mappings_path, 'r') as f:
            mappings = json.load(f)
            label2id = mappings['label2id']
            id2label = {int(k): v for k, v in mappings['id2label'].items()}
    else:
        label2id, id2label = create_label_mappings(test_labels)

    test_dataset = encode_data(tokenizer, test_texts, test_labels, label2id, MAX_LENGTH)

    # ── Step 4: Run evaluation ─────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Running evaluation...")
    print("="*60)

    eval_output_dir = './eval_output'
    training_args = TrainingArguments(
        output_dir=eval_output_dir,
        per_device_eval_batch_size=16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate(test_dataset)
    print(f"\nEvaluation Results ({source_label}):")
    print(f"  Loss:      {eval_results['eval_loss']:.4f}")
    print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")
    print(f"  F1:        {eval_results['eval_f1']:.4f}")

    # ── Step 5: Get predictions & classification report ────────
    print("\n" + "="*60)
    print("STEP 5: Detailed Classification Report")
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
    prefix = source_label  # 'local' or 'hub'

    plot_confusion_matrix(
        test_labels, predicted_labels, class_names,
        os.path.join(PLOTS_DIR, f'{prefix}_confusion_matrix.png')
    )
    plot_per_class_metrics(
        test_labels, predicted_labels, class_names,
        os.path.join(PLOTS_DIR, f'{prefix}_per_class_metrics.png')
    )
    plot_overall_metrics(
        eval_results,
        os.path.join(PLOTS_DIR, f'{prefix}_overall_metrics.png')
    )

    # ── Step 7: Save results ───────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, results_filename)

    results_to_save = {
        'source': source_label,
        'model_path': model_path,
        'eval_loss': eval_results['eval_loss'],
        'eval_accuracy': eval_results['eval_accuracy'],
        'eval_precision': eval_results['eval_precision'],
        'eval_recall': eval_results['eval_recall'],
        'eval_f1': eval_results['eval_f1'],
        'classification_report': report,
    }

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Clean up Trainer temp directory
    if os.path.exists(eval_output_dir):
        shutil.rmtree(eval_output_dir, ignore_errors=True)

    return results_to_save


if __name__ == "__main__":
    evaluate()
