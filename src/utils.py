"""
Utility module for model training and evaluation.

Provides the custom PyTorch Dataset class, label encoding helpers,
metrics computation, and data encoding functions.
"""

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class MyDataset(torch.utils.data.Dataset):
    """Custom PyTorch dataset for tokenized text data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_label_mappings(labels):
    """
    Create label-to-id and id-to-label mappings.

    Args:
        labels: List of string labels.

    Returns:
        Tuple of (label2id dict, id2label dict).
    """
    unique_labels = sorted(set(labels))  # sorted for reproducibility
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def compute_metrics(pred):
    """
    Compute evaluation metrics for the HuggingFace Trainer.

    Args:
        pred: Prediction output from the Trainer.

    Returns:
        Dictionary with accuracy, precision, recall, and f1 scores.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def encode_data(tokenizer, texts, labels, label2id, max_length=512):
    """
    Tokenize texts and encode labels into a PyTorch Dataset.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        texts: List of text strings.
        labels: List of label strings.
        label2id: Dictionary mapping labels to integer IDs.
        max_length: Maximum token length for truncation/padding.

    Returns:
        MyDataset instance ready for training/evaluation.
    """
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    encoded_labels = [label2id[y] for y in labels]
    return MyDataset(encodings, encoded_labels)
