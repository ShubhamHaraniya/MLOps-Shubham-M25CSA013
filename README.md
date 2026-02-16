# 📚 Goodreads Genre Classification — End-to-End MLOps Pipeline

> Fine-tuning **DistilBERT** on Goodreads book reviews for multi-genre classification, with full Docker deployment and HuggingFace Hub integration.

**Shubham Haraniya** · M25CSA013 · MLOps Assignment 3

[![HuggingFace](https://img.shields.io/badge/🤗_Model-spidey1807%2Fmlops__ass3-yellow)](https://huggingface.co/spidey1807/mlops_ass3)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://docker.com)

---

## 🎯 Overview

This project implements a complete MLOps pipeline that:

1. **Streams** book review data from the UCSD Goodreads dataset (8 genres, 16K reviews)
2. **Fine-tunes** a `distilbert-base-uncased` model for genre classification
3. **Evaluates** locally with detailed metrics and visualizations
4. **Pushes** the trained model to HuggingFace Hub
5. **Re-evaluates** from HuggingFace Hub to verify deployment integrity
6. **Containerizes** everything with GPU-enabled Docker images

### Genres Classified (8 classes)

| Genre | Genre | Genre | Genre |
|---|---|---|---|
| 📖 Children | 🎨 Comics & Graphic | 🧙 Fantasy & Paranormal | 📜 History & Biography |
| 🔍 Mystery/Thriller/Crime | ✍️ Poetry | 💕 Romance | 🌟 Young Adult |

---

## 📊 Results

| Metric | Score |
|---|---|
| **Accuracy** | **0.6188** |
| **Precision** | **0.6202** |
| **Recall** | **0.6188** |
| **F1-Score** | **0.6191** |

> Random baseline for 8 classes = 12.5% → Our model achieves **~5x improvement**

### Per-Class Performance

| Genre | Precision | Recall | F1-Score |
|---|---|---|---|
| Children | 0.71 | 0.70 | 0.71 |
| Comics & Graphic | 0.82 | 0.80 | **0.81** |
| Fantasy & Paranormal | 0.44 | 0.48 | 0.46 |
| History & Biography | 0.61 | 0.58 | 0.60 |
| Mystery/Thriller/Crime | 0.61 | 0.59 | 0.60 |
| Poetry | 0.76 | 0.78 | **0.77** |
| Romance | 0.57 | 0.61 | 0.59 |
| Young Adult | 0.43 | 0.40 | 0.41 |

### Local vs Hub Comparison

All metrics show **0.0000 difference** between local and hub models — confirming correct model upload/download.

---

## 🗂️ Project Structure

```
├── src/
│   ├── data.py              # Data streaming & preprocessing
│   ├── utils.py             # Dataset class, label encoding, metrics
│   ├── train.py             # DistilBERT training pipeline
│   ├── eval.py              # Evaluation with visualizations
│   └── push_to_hub.py       # Push model to HuggingFace Hub
├── evaluate_from_hub.py      # Re-evaluate from HF Hub
├── main.py                   # CLI entry point (full pipeline)
├── Dockerfile                # Dev Docker image (GPU)
├── Dockerfile.eval           # Production eval Docker image
├── requirements.txt          # Dependencies
├── M25CSA013_Assignment3.pdf # Report
├── results/
│   ├── eval_results.json     # Local evaluation results
│   ├── hub_eval_results.json # Hub evaluation results
│   ├── training_metrics.json # Training metrics
│   └── plots/                # Generated charts
│       ├── local_confusion_matrix.png
│       ├── local_per_class_metrics.png
│       ├── local_overall_metrics.png
│       ├── hub_confusion_matrix.png
│       ├── hub_per_class_metrics.png
│       ├── hub_overall_metrics.png
│       └── local_vs_hub_comparison.png
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- HuggingFace account with write-access token

### Installation
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main.py --all --repo_name spidey1807/mlops_ass3 --token <your-hf-token>
```

This runs: **Train → Evaluate → Push to Hub → Re-evaluate from Hub** in one command.

### Individual Steps

```bash
# Train the model
python main.py --train

# Evaluate locally
python main.py --eval

# Push to HuggingFace Hub
python main.py --push --repo_name spidey1807/mlops_ass3 --token <token>

# Re-evaluate from Hub
python main.py --eval_hub --repo_name spidey1807/mlops_ass3
```

---

## 🐳 Docker

### Development Image (Full Pipeline)
```bash
# Build
docker build -t mlops-ass3 .

# Train with GPU
docker run --gpus all mlops-ass3 python main.py --train

# Full pipeline
docker run --gpus all mlops-ass3 python main.py --all \
    --repo_name spidey1807/mlops_ass3 --token <token>
```

### Production Eval Image (Hub Only)
```bash
# Build
docker build -f Dockerfile.eval -t mlops-ass3-eval .

# Run evaluation
docker run --gpus all \
    -e HF_TOKEN=<your-token> \
    -e HF_REPO=spidey1807/mlops_ass3 \
    mlops-ass3-eval
```

---

## ⚙️ Model & Training Configuration

### Model: `distilbert-base-uncased`

| Property | Value |
|---|---|
| Parameters | 66M |
| Architecture | 6 layers, 768 hidden, 12 heads |
| Variant | Uncased |
| Task Head | SequenceClassification (8 labels) |

**Why DistilBERT?**
- Retains **97%** of BERT's language understanding at **60% faster** inference
- Good balance between accuracy and training speed (~20 min on GPU)
- Natively supports `AutoModelForSequenceClassification`

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Train Batch Size | 8 |
| Eval Batch Size | 16 |
| Learning Rate | 5e-5 |
| Warmup Steps | 100 |
| Weight Decay | 0.01 |
| Max Token Length | 256 |
| Optimizer | AdamW |
| Training Time | ~20.6 min (GPU) |

---

## 📈 Evaluation Visualizations

The pipeline auto-generates 7 evaluation plots in `results/plots/`:

| Plot | Description |
|---|---|
| `local_confusion_matrix.png` | Normalized confusion matrix (local model) |
| `local_per_class_metrics.png` | Per-genre precision, recall, F1 bar chart |
| `local_overall_metrics.png` | Overall metrics bar chart |
| `hub_confusion_matrix.png` | Confusion matrix (hub model) |
| `hub_per_class_metrics.png` | Per-genre metrics (hub model) |
| `hub_overall_metrics.png` | Overall metrics (hub model) |
| `local_vs_hub_comparison.png` | Side-by-side model comparison |

---

## 🔗 Links

| Resource | URL |
|---|---|
| 🤗 HuggingFace Model | [spidey1807/mlops_ass3](https://huggingface.co/spidey1807/mlops_ass3) |
| 📦 GitHub Repository | [ShubhamHaraniya/MLOps-Shubham-M25CSA013](https://github.com/ShubhamHaraniya/MLOps-Shubham-M25CSA013/tree/Assignment-3) |

---

## 📚 References

- [UCSD Goodreads Book Graph](https://mengtingwan.github.io/data/goodreads.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [DistilBERT Paper (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108)
- Original notebook by Maria Antoniak, Melanie Walsh, and the [AI for Humanists](https://aiforhumanists.com/) team


