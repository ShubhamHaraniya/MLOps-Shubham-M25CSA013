"""
Training module for fine-tuning a DistilBERT model on Goodreads genre classification.

Model Selection: distilbert-base-uncased
- Lightweight BERT variant with 66M parameters
- 6 hidden layers, 768 hidden size — good balance of speed and accuracy
- Suitable for sequence classification tasks with the Trainer API
- Trains in ~20 minutes on GPU
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.data import load_all_genres, split_data
from src.utils import create_label_mappings, encode_data, compute_metrics


# ── Configuration ──────────────────────────────────────────────
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 256
OUTPUT_DIR = './saved_model'
RESULTS_DIR = './results'
LOGGING_DIR = './logs'

# Training hyperparameters
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100


def train(
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    num_epochs=NUM_EPOCHS,
    device=None,
):
    """
    Full training pipeline: load data, prepare model, train, and save.

    Args:
        model_name: Pre-trained model identifier from HuggingFace.
        output_dir: Directory to save the fine-tuned model.
        num_epochs: Number of training epochs.
        device: Device to use ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        Tuple of (trainer, label2id, id2label, test_dataset, test_labels).
    """
    # Disable wandb logging
    os.environ["WANDB_DISABLED"] = "true"

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Loading Goodreads review data...")
    print("="*60)
    genre_reviews = load_all_genres(head=10000, sample_size=2000)
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)

    # ── Step 2: Prepare tokenizer and label mappings ───────────
    print("\n" + "="*60)
    print("STEP 2: Preparing tokenizer and label mappings...")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id, id2label = create_label_mappings(train_labels)
    print(f"Labels: {label2id}")

    # Save label mappings for later use
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'label_mappings.json'), 'w') as f:
        json.dump({'label2id': label2id, 'id2label': {str(k): v for k, v in id2label.items()}}, f, indent=2)

    # ── Step 3: Encode data ────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Encoding data for BERT...")
    print("="*60)
    train_dataset = encode_data(tokenizer, train_texts, train_labels, label2id, MAX_LENGTH)
    test_dataset = encode_data(tokenizer, test_texts, test_labels, label2id, MAX_LENGTH)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size:  {len(test_dataset)}")

    # ── Step 4: Load pre-trained model ─────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Loading pre-trained DistilBERT model...")
    print("="*60)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # ── Step 5: Configure training ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 5: Configuring training arguments...")
    print("="*60)
    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        output_dir='./training_output',
        logging_dir=LOGGING_DIR,
        logging_steps=LOGGING_STEPS,
        eval_strategy='steps',
        save_strategy='epoch',
        load_best_model_at_end=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ── Step 6: Train ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: Fine-tuning DistilBERT...")
    print("="*60)
    train_result = trainer.train()

    # Log training metrics
    train_metrics = train_result.metrics
    print(f"\nTraining complete!")
    print(f"  Training loss: {train_metrics['train_loss']:.4f}")
    print(f"  Training runtime: {train_metrics['train_runtime']:.1f}s")

    # ── Step 7: Save model ─────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7: Saving fine-tuned model...")
    print("="*60)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")

    # Save training metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Training metrics saved to: {metrics_path}")

    return trainer, label2id, id2label, test_dataset, test_labels


if __name__ == "__main__":
    train()