# ML-DL-Ops Assignment 1

This README contains the requested tables and results for the assignment.

## Q1 (A): Hyperparameter Tuning Results

| Dataset | BS | Opt | LR | PinMem | Epochs | Model | Test Acc | Best Val | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | 16 | SGD | 0.001 | True | 3 | ResNet-18 | 99.02% | 99.10% | 143.1s |
| MNIST | 16 | SGD | 0.001 | True | 3 | ResNet-50 | 98.48% | 98.40% | 226.9s |
| MNIST | 16 | SGD | 0.0001 | True | 3 | ResNet-18 | 98.31% | 98.40% | 142.5s |
| MNIST | 16 | SGD | 0.0001 | True | 3 | ResNet-50 | 97.06% | 96.77% | 227.4s |
| MNIST | 16 | Adam | 0.001 | True | 3 | ResNet-18 | 99.11% | 99.14% | 147.8s |
| MNIST | 16 | Adam | 0.001 | True | 3 | ResNet-50 | 97.91% | 98.47% | 242.4s |
| MNIST | 16 | Adam | 0.0001 | True | 3 | ResNet-18 | 98.87% | 98.84% | 147.5s |
| MNIST | 16 | Adam | 0.0001 | True | 3 | ResNet-50 | 98.34% | 98.51% | 243.2s |
| MNIST | 32 | SGD | 0.001 | True | 3 | ResNet-18 | 98.94% | 99.09% | 104.7s |
| MNIST | 32 | SGD | 0.001 | True | 3 | ResNet-50 | 98.71% | 98.60% | 146.8s |
| MNIST | 32 | Adam | 0.0001 | True | 3 | ResNet-18 | 98.60% | 98.67% | 109.2s |
| MNIST | 32 | Adam | 0.0001 | True | 3 | ResNet-50 | 98.24% | 98.24% | 154.2s |
| MNIST | 16 | SGD | 0.001 | True | 5 | ResNet-18 | 99.33% | 99.30% | 235.0s |
| MNIST | 16 | SGD | 0.001 | True | 5 | ResNet-50 | 98.78% | 98.81% | 372.7s |
| MNIST | 16 | SGD | 0.0001 | True | 5 | ResNet-18 | 98.47% | 98.56% | 234.2s |
| MNIST | 16 | SGD | 0.0001 | True | 5 | ResNet-50 | 97.98% | 98.34% | 374.0s |
| MNIST | 16 | Adam | 0.001 | True | 5 | ResNet-18 | 99.36% | 99.31% | 242.9s |
| MNIST | 16 | Adam | 0.001 | True | 5 | ResNet-50 | 98.68% | 98.99% | 399.3s |
| MNIST | 16 | Adam | 0.0001 | True | 5 | ResNet-18 | 99.32% | 99.30% | 241.7s |
| MNIST | 16 | Adam | 0.0001 | True | 5 | ResNet-50 | 98.76% | 98.87% | 399.0s |
| MNIST | 32 | SGD | 0.001 | True | 5 | ResNet-18 | 99.16% | 99.06% | 169.5s |
| MNIST | 32 | SGD | 0.001 | True | 5 | ResNet-50 | 98.92% | 98.86% | 238.8s |
| MNIST | 32 | Adam | 0.0001 | True | 5 | ResNet-18 | 98.99% | 99.07% | 177.5s |
| MNIST | 32 | Adam | 0.0001 | True | 5 | ResNet-50 | 98.59% | 98.67% | 252.6s |
| MNIST | 16 | SGD | 0.001 | False | 3 | ResNet-18 | 99.10% | 98.69% | 148.2s |
| MNIST | 16 | SGD | 0.001 | False | 3 | ResNet-50 | 98.73% | 98.59% | 229.1s |
| MNIST | 16 | SGD | 0.0001 | False | 3 | ResNet-18 | 98.36% | 98.29% | 143.8s |
| MNIST | 16 | SGD | 0.0001 | False | 3 | ResNet-50 | 97.26% | 97.34% | 228.3s |
| MNIST | 16 | Adam | 0.001 | False | 3 | ResNet-18 | 99.04% | 99.00% | 149.2s |
| MNIST | 16 | Adam | 0.001 | False | 3 | ResNet-50 | 98.59% | 98.61% | 243.8s |
| MNIST | 16 | Adam | 0.0001 | False | 3 | ResNet-18 | 98.70% | 99.01% | 149.3s |
| MNIST | 16 | Adam | 0.0001 | False | 3 | ResNet-50 | 98.01% | 98.14% | 243.9s |
| MNIST | 32 | SGD | 0.001 | False | 3 | ResNet-18 | 98.96% | 99.10% | 104.9s |
| MNIST | 32 | SGD | 0.001 | False | 3 | ResNet-50 | 98.66% | 98.53% | 151.9s |
| MNIST | 32 | Adam | 0.0001 | False | 3 | ResNet-18 | 98.79% | 98.79% | 108.5s |
| MNIST | 32 | Adam | 0.0001 | False | 3 | ResNet-50 | 98.60% | 98.33% | 155.0s |
| MNIST | 16 | SGD | 0.001 | False | 5 | ResNet-18 | 99.09% | 99.11% | 235.9s |
| MNIST | 16 | SGD | 0.001 | False | 5 | ResNet-50 | 99.05% | 99.07% | 375.0s |
| MNIST | 16 | SGD | 0.0001 | False | 5 | ResNet-18 | 98.61% | 98.80% | 234.5s |
| MNIST | 16 | SGD | 0.0001 | False | 5 | ResNet-50 | 98.21% | 98.37% | 374.9s |
| MNIST | 16 | Adam | 0.001 | False | 5 | ResNet-18 | 99.16% | 99.09% | 243.5s |
| MNIST | 16 | Adam | 0.001 | False | 5 | ResNet-50 | 98.80% | 98.81% | 398.9s |
| MNIST | 16 | Adam | 0.0001 | False | 5 | ResNet-18 | 99.15% | 99.20% | 243.6s |
| MNIST | 16 | Adam | 0.0001 | False | 5 | ResNet-50 | 98.74% | 98.80% | 400.9s |
| MNIST | 32 | SGD | 0.001 | False | 5 | ResNet-18 | 99.06% | 99.07% | 170.8s |
| MNIST | 32 | SGD | 0.001 | False | 5 | ResNet-50 | 98.72% | 98.89% | 240.3s |
| MNIST | 32 | Adam | 0.0001 | False | 5 | ResNet-18 | 98.89% | 99.01% | 175.9s |
| MNIST | 32 | Adam | 0.0001 | False | 5 | ResNet-50 | 98.69% | 98.89% | 251.8s |
| FashionMNIST | 16 | SGD | 0.001 | True | 3 | ResNet-18 | 90.18% | 90.44% | 144.0s |
| FashionMNIST | 16 | SGD | 0.001 | True | 3 | ResNet-50 | 89.49% | 89.11% | 227.8s |
| FashionMNIST | 16 | SGD | 0.0001 | True | 3 | ResNet-18 | 89.05% | 89.03% | 143.4s |
| FashionMNIST | 16 | SGD | 0.0001 | True | 3 | ResNet-50 | 85.04% | 84.53% | 229.5s |
| FashionMNIST | 16 | Adam | 0.001 | True | 3 | ResNet-18 | 90.93% | 90.67% | 148.8s |
| FashionMNIST | 16 | Adam | 0.001 | True | 3 | ResNet-50 | 88.55% | 88.64% | 243.6s |
| FashionMNIST | 16 | Adam | 0.0001 | True | 3 | ResNet-18 | 90.45% | 90.83% | 148.5s |
| FashionMNIST | 16 | Adam | 0.0001 | True | 3 | ResNet-50 | 87.74% | 88.69% | 243.1s |
| FashionMNIST | 32 | SGD | 0.001 | True | 3 | ResNet-18 | 90.64% | 89.97% | 105.3s |
| FashionMNIST | 32 | SGD | 0.001 | True | 3 | ResNet-50 | 88.55% | 87.87% | 146.6s |
| FashionMNIST | 32 | Adam | 0.0001 | True | 3 | ResNet-18 | 91.54% | 91.26% | 109.0s |
| FashionMNIST | 32 | Adam | 0.0001 | True | 3 | ResNet-50 | 88.39% | 88.79% | 154.2s |
| FashionMNIST | 16 | SGD | 0.001 | True | 5 | ResNet-18 | 91.56% | 91.97% | 233.1s |
| FashionMNIST | 16 | SGD | 0.001 | True | 5 | ResNet-50 | 90.55% | 90.74% | 373.9s |
| FashionMNIST | 16 | SGD | 0.0001 | True | 5 | ResNet-18 | 89.96% | 90.16% | 234.2s |
| FashionMNIST | 16 | SGD | 0.0001 | True | 5 | ResNet-50 | 87.08% | 86.89% | 373.7s |
| FashionMNIST | 16 | Adam | 0.001 | True | 5 | ResNet-18 | 91.49% | 92.03% | 241.6s |
| FashionMNIST | 16 | Adam | 0.001 | True | 5 | ResNet-50 | 91.09% | 91.13% | 398.1s |
| FashionMNIST | 16 | Adam | 0.0001 | True | 5 | ResNet-18 | 91.56% | 91.73% | 241.7s |
| FashionMNIST | 16 | Adam | 0.0001 | True | 5 | ResNet-50 | 91.31% | 91.50% | 399.0s |
| FashionMNIST | 32 | SGD | 0.001 | True | 5 | ResNet-18 | 91.49% | 91.12% | 170.5s |
| FashionMNIST | 32 | SGD | 0.001 | True | 5 | ResNet-50 | 89.28% | 89.44% | 240.2s |
| FashionMNIST | 32 | Adam | 0.0001 | True | 5 | ResNet-18 | 92.05% | 91.88% | 177.1s |
| FashionMNIST | 32 | Adam | 0.0001 | True | 5 | ResNet-50 | 89.50% | 89.76% | 253.9s |
| FashionMNIST | 16 | SGD | 0.001 | False | 3 | ResNet-18 | 90.15% | 90.38% | 144.5s |
| FashionMNIST | 16 | SGD | 0.001 | False | 3 | ResNet-50 | 89.35% | 89.05% | 228.1s |
| FashionMNIST | 16 | SGD | 0.0001 | False | 3 | ResNet-18 | 89.12% | 89.08% | 144.1s |
| FashionMNIST | 16 | SGD | 0.0001 | False | 3 | ResNet-50 | 85.11% | 84.67% | 230.2s |
| FashionMNIST | 16 | Adam | 0.001 | False | 3 | ResNet-18 | 91.01% | 90.72% | 149.3s |
| FashionMNIST | 16 | Adam | 0.001 | False | 3 | ResNet-50 | 88.62% | 88.58% | 244.0s |
| FashionMNIST | 16 | Adam | 0.0001 | False | 3 | ResNet-18 | 90.55% | 90.79% | 149.0s |
| FashionMNIST | 16 | Adam | 0.0001 | False | 3 | ResNet-50 | 87.89% | 88.75% | 243.8s |
| FashionMNIST | 32 | SGD | 0.001 | False | 3 | ResNet-18 | 90.66% | 90.02% | 106.1s |
| FashionMNIST | 32 | SGD | 0.001 | False | 3 | ResNet-50 | 88.60% | 87.94% | 147.2s |
| FashionMNIST | 32 | Adam | 0.0001 | False | 3 | ResNet-18 | 91.49% | 91.31% | 109.8s |
| FashionMNIST | 32 | Adam | 0.0001 | False | 3 | ResNet-50 | 88.45% | 88.85% | 154.9s |
| FashionMNIST | 16 | SGD | 0.001 | False | 5 | ResNet-18 | 91.60% | 91.95% | 234.5s |
| FashionMNIST | 16 | SGD | 0.001 | False | 5 | ResNet-50 | 90.62% | 90.81% | 375.1s |
| FashionMNIST | 16 | SGD | 0.0001 | False | 5 | ResNet-18 | 90.05% | 90.22% | 235.1s |
| FashionMNIST | 16 | SGD | 0.0001 | False | 5 | ResNet-50 | 87.15% | 87.01% | 374.8s |
| FashionMNIST | 16 | Adam | 0.001 | False | 5 | ResNet-18 | 91.55% | 92.10% | 242.8s |
| FashionMNIST | 16 | Adam | 0.001 | False | 5 | ResNet-50 | 91.15% | 91.20% | 399.5s |
| FashionMNIST | 16 | Adam | 0.0001 | False | 5 | ResNet-18 | 91.61% | 91.79% | 242.9s |
| FashionMNIST | 16 | Adam | 0.0001 | False | 5 | ResNet-50 | 91.38% | 91.56% | 400.2s |
| FashionMNIST | 32 | SGD | 0.001 | False | 5 | ResNet-18 | 91.45% | 91.08% | 172.1s |
| FashionMNIST | 32 | SGD | 0.001 | False | 5 | ResNet-50 | 89.34% | 89.51% | 241.6s |
| FashionMNIST | 32 | Adam | 0.0001 | False | 5 | ResNet-18 | 91.95% | 91.75% | 178.6s |
| FashionMNIST | 32 | Adam | 0.0001 | False | 5 | ResNet-50 | 89.62% | 89.81% | 254.3s |

## Q1 (B): SVM Results

| Dataset | Kernel | Test Acc (%) | Time (ms) |
|---|---|---|---|
| MNIST | poly | 97.71% | 233846.10ms |
| MNIST | rbf | 97.92% | 272000.61ms |
| FashionMNIST | poly | 86.30% | 393137.24ms |
| FashionMNIST | rbf | 88.29% | 397992.41ms |

## Q2: Benchmarking Results

### CPU Results
| Device | Model | BS | Opt | LR | Train Time (ms) | FLOPs | Acc (%) |
|---|---|---|---|---|---|---|---|
| cpu | ResNet-18 | 16 | SGD | 0.001 | 135015.12 | 0.2961 | 87.65 |
| cpu | ResNet-50 | 16 | SGD | 0.001 | 346542.64 | 0.6673 | 85.63 |
| cpu | ResNet-18 | 16 | Adam | 0.001 | 165550.63 | 0.2961 | 88.19 |
| cpu | ResNet-50 | 16 | Adam | 0.001 | 420020.88 | 0.6673 | 87.13 |

### GPU Results
| Device | Model | BS | Opt | LR | Train Time (ms) | FLOPs | Acc (%) |
|---|---|---|---|---|---|---|---|
| cuda | ResNet-18 | 16 | SGD | 0.001 | 20336.38 | 0.2961 | 88.41 |
| cuda | ResNet-50 | 16 | SGD | 0.001 | 46177.53 | 0.6673 | 85.49 |
| cuda | ResNet-18 | 16 | Adam | 0.001 | 25460.17 | 0.2961 | 88.26 |
| cuda | ResNet-50 | 16 | Adam | 0.001 | 52566.23 | 0.6673 | 84.07 |
