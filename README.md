# MLOps Assignment 5 — M25CSA013 | Shubham Haraniya

> **Branch:** `Assignment 5` | **Roll No:** M25CSA013

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🔗 WandB Project | [mlops-assignment5](https://wandb.ai/spharaniya18-intelkit-solutions/mlops-assignment5) |
| 🤗 HuggingFace Model (Q1 Best) | [spidey1807/vit-small-cifar100-lora](https://huggingface.co/spidey1807/vit-small-cifar100-lora) |

---

## Project Structure

```
Assignment 5/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── q1_vit_lora/
│   ├── config.py          # all 10 experiment configs
│   ├── dataset.py         # CIFAR-100 dataloaders (224×224, ImageNet norm)
│   ├── model.py           # ViT-S with / without LoRA (PEFT)
│   ├── train.py           # training + WandB logging (loss, acc, grad norms)
│   ├── test.py            # test accuracy + class-wise histogram
│   ├── optuna_search.py   # Optuna HPO for LoRA hyperparameters
│   └── push_to_hf.py      # push best model to HuggingFace
└── q2_adversarial/
    ├── config.py
    ├── dataset.py
    ├── resnet_model.py         # ResNet-18 (classifier) + ResNet-34 (detector)
    ├── train_resnet18.py       # train on clean CIFAR-10 from scratch
    ├── fgsm_scratch.py         # FGSM implemented without ART
    ├── fgsm_art.py             # FGSM using IBM ART + visual comparison
    ├── train_detector.py       # ResNet-34 detectors for PGD and BIM
    ├── evaluate_detector.py    # confusion matrix, AUC, comparison chart
    └── visualize.py            # image grids + WandB sample logging
```

---

## Installation

### Option A — Without Docker (Direct Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set credentials (Windows CMD)
set WANDB_API_KEY=your_api_key_here
set HF_TOKEN=your_hf_token_here

# Or copy .env.example to .env and fill in your keys (Linux/Mac)
cp .env.example .env && source .env
```

### Option B — Docker (also supported by assignment)

```bash
docker compose build
docker compose run --rm mlops bash
# Then inside container, set env vars and run scripts normally
```

---

## ▶️ Run Everything (one command)

```bash
# From the project root — runs full Q1 + Q2 pipeline
python run_pipeline.py --device cuda

# Individual options:
python run_pipeline.py --q1 --device cuda           # Q1 only
python run_pipeline.py --q2 --device cuda           # Q2 only
python run_pipeline.py --q1 --exp 5 --device cuda   # single Q1 experiment
python run_pipeline.py --q1 --optuna --push_hf      # Q1 + Optuna + HF push
python run_pipeline.py --device cpu                  # CPU fallback (slow)
```

---

## Q1 — ViT-S Fine-tuning on CIFAR-100

### How to Run

```bash
# All 10 experiments (exp 0 = no-LoRA baseline; exps 1-9 = LoRA grid)
cd q1_vit_lora
python train.py --exp -1 --device cuda

# Single experiment (e.g., exp 5: rank=4, alpha=4)
python train.py --exp 5 --device cuda

# Test all checkpoints
python test.py --exp -1 --device cuda

# Optuna hyperparameter search (20 trials) + retrain best config
python optuna_search.py --trials 20 --retrain --device cuda

# Push best model to HuggingFace Hub
python push_to_hf.py
```

---

### Q1 Experiment Grid

| Exp | LoRA | Rank | Alpha | Dropout |
|-----|------|------|-------|---------|
| 0   | No   | —    | —     | —       |
| 1   | Yes  | 2    | 2     | 0.1     |
| 2   | Yes  | 2    | 4     | 0.1     |
| 3   | Yes  | 2    | 8     | 0.1     |
| 4   | Yes  | 4    | 2     | 0.1     |
| 5   | Yes  | 4    | 4     | 0.1     |
| 6   | Yes  | 4    | 8     | 0.1     |
| 7   | Yes  | 8    | 2     | 0.1     |
| 8   | Yes  | 8    | 4     | 0.1     |
| 9   | Yes  | 8    | 8     | 0.1     |

---

### Q1 Train-Val Tables (per experiment, 10 epochs)

> **Experiment 0 — Baseline (No LoRA), Rank: —, Alpha: —**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 2.5700        | 1.4392          | 0.4756            | 0.6852              |
| 2     | 1.1569        | 1.0522          | 0.7286            | 0.7346              |
| 3     | 0.9358        | 0.9341          | 0.7568            | 0.7536              |
| 4     | 0.8423        | 0.8818          | 0.7722            | 0.7596              |
| 5     | 0.7916        | 0.8497          | 0.7816            | 0.7660              |
| 6     | 0.7624        | 0.8304          | 0.7875            | 0.7728              |
| 7     | 0.7369        | 0.8198          | 0.7942            | 0.7722              |
| 8     | 0.7274        | 0.8156          | 0.7960            | 0.7718              |
| 9     | 0.7152        | 0.8128          | 0.7999            | 0.7740              |
| 10    | 0.7072        | 0.8120          | 0.8023            | 0.7746              |

> **Experiment 1 — LoRA, Rank: 2, Alpha: 2**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 2.0466        | 0.7452          | 0.5750            | 0.8172              |
| 2     | 0.5370        | 0.5119          | 0.8525            | 0.8536              |
| 3     | 0.4185        | 0.4530          | 0.8767            | 0.8642              |
| 4     | 0.3710        | 0.4259          | 0.8864            | 0.8678              |
| 5     | 0.3436        | 0.4068          | 0.8944            | 0.8768              |
| 6     | 0.3212        | 0.3979          | 0.9004            | 0.8770              |
| 7     | 0.3063        | 0.3889          | 0.9040            | 0.8806              |
| 8     | 0.2973        | 0.3879          | 0.9088            | 0.8804              |
| 9     | 0.2924        | 0.3855          | 0.9092            | 0.8814              |
| 10    | 0.2884        | 0.3851          | 0.9106            | 0.8816              |

> **Experiment 2 — LoRA, Rank: 2, Alpha: 4**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 1.9239        | 0.6911          | 0.5953            | 0.8268              |
| 2     | 0.5147        | 0.4853          | 0.8580            | 0.8570              |
| 3     | 0.4073        | 0.4246          | 0.8791            | 0.8712              |
| 4     | 0.3607        | 0.4037          | 0.8901            | 0.8776              |
| 5     | 0.3305        | 0.3939          | 0.8981            | 0.8808              |
| 6     | 0.3106        | 0.3838          | 0.9053            | 0.8794              |
| 7     | 0.2965        | 0.3769          | 0.9087            | 0.8844              |
| 8     | 0.2866        | 0.3692          | 0.9111            | 0.8860              |
| 9     | 0.2791        | 0.3702          | 0.9128            | 0.8854              |
| 10    | 0.2794        | 0.3690          | 0.9129            | 0.8850              |

> **Experiment 3 — LoRA, Rank: 2, Alpha: 8**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 1.8228        | 0.6677          | 0.6131            | 0.8276              |
| 2     | 0.5021        | 0.4809          | 0.8593            | 0.8558              |
| 3     | 0.3989        | 0.4332          | 0.8812            | 0.8698              |
| 4     | 0.3517        | 0.4047          | 0.8917            | 0.8736              |
| 5     | 0.3238        | 0.3847          | 0.9001            | 0.8814              |
| 6     | 0.3004        | 0.3783          | 0.9064            | 0.8820              |
| 7     | 0.2883        | 0.3717          | 0.9109            | 0.8846              |
| 8     | 0.2767        | 0.3666          | 0.9132            | 0.8844              |
| 9     | 0.2708        | 0.3636          | 0.9157            | 0.8860              |
| 10    | 0.2649        | 0.3631          | 0.9171            | 0.8864              |

> **Experiment 4 — LoRA, Rank: 4, Alpha: 2**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 2.0438        | 0.7332          | 0.5743            | 0.8290              |
| 2     | 0.5326        | 0.5120          | 0.8549            | 0.8590              |
| 3     | 0.4170        | 0.4502          | 0.8760            | 0.8680              |
| 4     | 0.3688        | 0.4161          | 0.8876            | 0.8758              |
| 5     | 0.3389        | 0.4028          | 0.8948            | 0.8780              |
| 6     | 0.3162        | 0.3961          | 0.9011            | 0.8780              |
| 7     | 0.3043        | 0.3882          | 0.9060            | 0.8840              |
| 8     | 0.2955        | 0.3850          | 0.9089            | 0.8840              |
| 9     | 0.2882        | 0.3832          | 0.9111            | 0.8844              |
| 10    | 0.2895        | 0.3827          | 0.9099            | 0.8844              |

> **Experiment 5 — LoRA, Rank: 4, Alpha: 4**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 1.9428        | 0.6768          | 0.5919            | 0.8280              |
| 2     | 0.5129        | 0.4843          | 0.8558            | 0.8544              |
| 3     | 0.4071        | 0.4323          | 0.8789            | 0.8712              |
| 4     | 0.3583        | 0.3970          | 0.8907            | 0.8808              |
| 5     | 0.3285        | 0.3881          | 0.8972            | 0.8816              |
| 6     | 0.3065        | 0.3785          | 0.9045            | 0.8834              |
| 7     | 0.2935        | 0.3668          | 0.9090            | 0.8848              |
| 8     | 0.2851        | 0.3651          | 0.9116            | 0.8864              |
| 9     | 0.2787        | 0.3634          | 0.9132            | 0.8858              |
| 10    | 0.2740        | 0.3625          | 0.9137            | 0.8858              |

> **Experiment 6 — LoRA, Rank: 4, Alpha: 8**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 1.7953        | 0.6452          | 0.6192            | 0.8364              |
| 2     | 0.4953        | 0.4793          | 0.8598            | 0.8620              |
| 3     | 0.3919        | 0.4223          | 0.8815            | 0.8702              |
| 4     | 0.3455        | 0.3939          | 0.8934            | 0.8774              |
| 5     | 0.3172        | 0.3794          | 0.9022            | 0.8830              |
| 6     | 0.2963        | 0.3706          | 0.9074            | 0.8894              |
| 7     | 0.2798        | 0.3631          | 0.9126            | 0.8910              |
| 8     | 0.2691        | 0.3600          | 0.9158            | 0.8904              |
| 9     | 0.2629        | 0.3598          | 0.9178            | 0.8926              |
| 10    | 0.2609        | 0.3590          | 0.9194            | 0.8926              |

> **Experiment 7 — LoRA, Rank: 8, Alpha: 2**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 1.9894        | 0.7249          | 0.5833            | 0.8242              |
| 2     | 0.5353        | 0.5036          | 0.8522            | 0.8576              |
| 3     | 0.4208        | 0.4398          | 0.8751            | 0.8712              |
| 4     | 0.3706        | 0.4152          | 0.8882            | 0.8756              |
| 5     | 0.3404        | 0.3952          | 0.8959            | 0.8808              |
| 6     | 0.3188        | 0.3889          | 0.9005            | 0.8838              |
| 7     | 0.3070        | 0.3817          | 0.9034            | 0.8838              |
| 8     | 0.2970        | 0.3791          | 0.9087            | 0.8834              |
| 9     | 0.2902        | 0.3761          | 0.9106            | 0.8856              |
| 10    | 0.2869        | 0.3757          | 0.9107            | 0.8860              |

> **Experiment 8 — LoRA, Rank: 8, Alpha: 4**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 1.9479        | 0.6935          | 0.5910            | 0.8224              |
| 2     | 0.5104        | 0.4836          | 0.8587            | 0.8574              |
| 3     | 0.4021        | 0.4278          | 0.8794            | 0.8702              |
| 4     | 0.3548        | 0.3992          | 0.8924            | 0.8780              |
| 5     | 0.3261        | 0.3879          | 0.9003            | 0.8816              |
| 6     | 0.3056        | 0.3743          | 0.9047            | 0.8844              |
| 7     | 0.2901        | 0.3668          | 0.9102            | 0.8876              |
| 8     | 0.2793        | 0.3652          | 0.9128            | 0.8878              |
| 9     | 0.2747        | 0.3627          | 0.9142            | 0.8890              |
| 10    | 0.2699        | 0.3621          | 0.9154            | 0.8894              |

> **Experiment 9 — LoRA, Rank: 8, Alpha: 8**

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|-------------------|---------------------|
| 1     | 1.8860        | 0.6554          | 0.5999            | 0.8310              |
| 2     | 0.4966        | 0.4646          | 0.8603            | 0.8626              |
| 3     | 0.3921        | 0.4104          | 0.8820            | 0.8756              |
| 4     | 0.3462        | 0.3943          | 0.8937            | 0.8776              |
| 5     | 0.3152        | 0.3752          | 0.9008            | 0.8850              |
| 6     | 0.2933        | 0.3630          | 0.9081            | 0.8838              |
| 7     | 0.2787        | 0.3599          | 0.9121            | 0.8868              |
| 8     | 0.2676        | 0.3526          | 0.9164            | 0.8862              |
| 9     | 0.2601        | 0.3532          | 0.9188            | 0.8860              |
| 10    | 0.2575        | 0.3527          | 0.9193            | 0.8862              |

---

### Q1 Test Accuracy Table

| LoRA layers | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Parameters |
|-------------|------|-------|---------|-----------------------|----------------------|
| Without     | —    | —     | —       | 0.7657                | 38,500               |
| With        | 2    | 2     | 0.1     | 0.8856                | 75,364               |
| With        | 2    | 4     | 0.1     | 0.8867                | 75,364               |
| With        | 2    | 8     | 0.1     | 0.8910                | 75,364               |
| With        | 4    | 2     | 0.1     | 0.8867                | 112,228              |
| With        | 4    | 4     | 0.1     | 0.8881                | 112,228              |
| With        | 4    | 8     | 0.1     | 0.8915                | 112,228              |
| With        | 8    | 2     | 0.1     | 0.8874                | 185,956              |
| With        | 8    | 4     | 0.1     | 0.8902                | 185,956              |
| With        | 8    | 8     | 0.1     | 0.8905                | 185,956              |

---

### Q1 Optuna Best Hyperparameter Configuration

| Hyperparameter | Best Value |
|----------------|------------|
| Rank           | 16         |
| Alpha          | 16         |
| Dropout        | 0.25       |
| Learning Rate  | 4.93 × 10⁻⁴ |
| Val Accuracy (5-epoch search) | 0.9046 |
| Val Accuracy (10-epoch retrain) | 0.9020 |

---

## Q2 — Adversarial Attacks using IBM ART

### How to Run

```bash
cd q2_adversarial

# Step 1: Train ResNet-18 from scratch (target >= 72%)
python train_resnet18.py --epochs 30 --device cuda

# Step 2: FGSM from scratch (no ART)
python fgsm_scratch.py --device cuda

# Step 3: FGSM using IBM ART + comparison
python fgsm_art.py --device cuda

# Step 4: Train adversarial detectors (PGD + BIM)
python train_detector.py --attack both --device cuda --n_train 10000

# Step 5: Evaluate detectors
python evaluate_detector.py --attack both --device cuda

# Step 6: Generate all visualisations + log to WandB
python visualize.py --device cuda
```

---

### Q2(i) ResNet-18 Clean Training Table (30 epochs)

| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|-------|-----------|----------------|-----------|---------------|
| 1     | 1.9140     | 0.3197         | 1.6168    | 0.4029        |
| 2     | 1.4085     | 0.4869         | 1.2659    | 0.5437        |
| 3     | 1.1074     | 0.6033         | 1.1636    | 0.5889        |
| 4     | 0.8821     | 0.6875         | 0.8471    | 0.7072        |
| 5     | 0.7154     | 0.7504         | 0.9422    | 0.6996        |
| 10    | 0.4840     | 0.8341         | 0.6723    | 0.7819        |
| 15    | 0.4173     | 0.8552         | 0.6340    | 0.7955        |
| 20    | 0.1506     | 0.9479         | 0.2820    | 0.9074        |
| 25    | 0.0993     | 0.9659         | 0.2643    | 0.9170        |
| 30    | 0.0540     | 0.9835         | 0.2324    | **0.9312**    |

---

### Q2(i) FGSM Accuracy Comparison Table

| Method                  | Epsilon | Accuracy | Accuracy Drop vs Clean |
|-------------------------|---------|----------|------------------------|
| Clean (no attack)       | —       | 0.9312   | —                      |
| FGSM from Scratch       | 0.01    | 0.4631   | 0.4681                 |
| FGSM from Scratch       | 0.03    | 0.3917   | 0.5395                 |
| FGSM from Scratch       | 0.05    | 0.3485   | 0.5827                 |
| FGSM from Scratch       | 0.10    | 0.2834   | 0.6478                 |
| FGSM from Scratch       | 0.20    | 0.2197   | 0.7115                 |
| FGSM from Scratch       | 0.30    | 0.1870   | 0.7442                 |
| FGSM using IBM ART      | 0.01    | 0.2740   | 0.1735                 |
| FGSM using IBM ART      | 0.03    | 0.2385   | 0.2090                 |
| FGSM using IBM ART      | 0.05    | 0.1875   | 0.2600                 |
| FGSM using IBM ART      | 0.10    | 0.1385   | 0.3090                 |
| FGSM using IBM ART      | 0.20    | 0.1370   | 0.3105                 |
| FGSM using IBM ART      | 0.30    | 0.1355   | 0.3120                 |

---

### Q2(ii) Adversarial Detector Training Tables

> **PGD Detector — ResNet-34 (Binary: clean=0, adversarial=1)**

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| 1     | 0.5248         | 0.5519       |
| 2     | 0.7538         | 0.8219       |
| 5     | 0.9927         | 0.9689       |
| 10    | 0.9995         | 0.9825       |
| 15    | 1.0000         | 0.9842       |
| 20    | 1.0000         | **0.9858** (best: 0.9864) |

> **BIM Detector — ResNet-34 (Binary: clean=0, adversarial=1)**

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| 1     | 0.5371         | 0.6203       |
| 2     | 0.8521         | 0.9703       |
| 5     | 0.9959         | 0.9694       |
| 10    | 0.9983         | 0.9828       |
| 15    | 1.0000         | 0.9844       |
| 20    | 0.9999         | **0.9814** (best: 0.9850) |

---

### Q2(ii) Detector Performance Comparison

| Attack | Detection Accuracy | ROC-AUC | Target (≥70%) |
|--------|--------------------|---------|----------------|
| PGD    | 0.9975             | 1.0000  | ✓              |
| BIM    | 0.9978             | 0.9998  | ✓              |

---

## Notes

- All experiments run inside Docker container.
- WandB logs: train/val curves, LoRA gradient norm graphs, class-wise histogram (Q1); 10 clean+adversarial sample pairs per attack (Q2).
- Best Q1 model weights are in `weights/q1_best/` and pushed to HuggingFace.
- All Q2 model weights in `weights/q2_all/`.
