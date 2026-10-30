# Super-Resolution Generative Adversarial Network (SRGAN) for Cats vs Dogs

This project implements an **SRGAN (32 × 32 → 128 × 128)** and evaluates its effect on a binary Cats vs Dogs classification task. Two **PyTorch ResNet18** transfer-learning classifiers are compared: **Model A**, trained on original 128 × 128 images, and **Model B**, trained on SRGAN-generated 128 × 128 images.

## Table of Contents
1. [Introduction](#1-introduction)
2. [Project Structure](#2-project-structure)
3. [Getting Started](#3-getting-started)
4. [Training Process](#4-training-process)
5. [Results and Evaluation](#5-results-and-evaluation)
6. [Conclusions and Limitations](#6-conclusions-and-limitations)
7. [References](#7-references)

## 1. Introduction

Single-image super-resolution reconstructs a high-resolution image from a low-resolution input. The SRGAN generator learns to reconstruct a 128 × 128 RGB image from a 32 × 32 RGB input; the discriminator distinguishes generated images from reference images. The experiment tests whether reconstruction improvements translate into improved downstream Cats vs Dogs classification performance.

## 2. Project Structure

```text
SRGAN-Cats-Dogs-Midterm/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Applied_AI_midterm.ipynb
├── models/
│   └── README.md                  # Checkpoint instructions and model naming
├── images/                       # Exported figures from the training run
├── results/                      # CSV/JSON metrics and reports
├── docs/
│   └── reproduction_notes.md
└── data/
    └── README.md                  # Dataset acquisition and placement; no dataset committed
```

**The notebook itself produces** `SRGAN_Midterm/checkpoints/`, `SRGAN_Midterm/figures/`, and `SRGAN_Midterm/results/` **at runtime**. To publish the figures, copy selected files from `SRGAN_Midterm/figures/` into this repository's `images/`, and results from `SRGAN_Midterm/results/` into `results/`.

## 3. Getting Started

### Prerequisites

- Python 3.x and a Jupyter environment (GPU recommended)
- PyTorch and torchvision
- Dependencies in `requirements.txt`
- Cats vs Dogs images stored locally; raw dataset is not included in this repository

### Setup

```bash
git clone https://github.com/YOUR-USERNAME/SRGAN-Cats-Dogs-Midterm.git
cd SRGAN-Cats-Dogs-Midterm
python -m pip install -r requirements.txt
jupyter lab
```

Launch Jupyter **from the repository root**. The notebook assumes `PROJECT_ROOT = Path.cwd()` and expects the dataset at `data/dogs-vs-cats-classification` relative to that directory. If Jupyter starts in a different directory, update `PROJECT_ROOT` in the setup cells.

The notebook also expects labeled image names/folders (`cat.*`, `dog.*`, `cats/`, `dogs/`), or the same original dataset layout, and saves output into `SRGAN_Midterm/`. Make sure saved split CSV paths resolve on your machine; if moving the dataset, regenerate consistent splits before retraining or adapt the CSV paths. Do not alter splits between Model A and B.

## 4. Training Process

### Step 1: Classifier A

Original images are resized to **128 × 128**. A pretrained **ResNet18** is adapted with a two-class classification head and trained using transfer learning. A best validation checkpoint is saved as `classifier_A_latest_best.pth`.

### Step 2: SRGAN

- Generator: 8 residual blocks; two PixelShuffle upsampling stages (4× total)
- Input: **32 × 32 RGB** images; target/output: **128 × 128 RGB** images
- Discriminator: convolutional real-versus-generated classifier
- Generator objective: pixel MSE + VGG16 perceptual loss + adversarial BCE-with-logits loss
- Training: generator pretraining followed by **150 adversarial epochs**
- Resuming: `srgan_latest.pth` saves generator/discriminator states, optimizers, epoch/history, and mixed-precision scalers where present; periodic numbered checkpoints are also saved

### Step 3: Classifier B

The final generator is frozen. Training images are downscaled to 32 × 32, reconstructed to 128 × 128 through SRGAN, normalized for ResNet18, and fed to **a separate ResNet18 binary classifier** with the same architecture as Model A. This notebook does **not** train Model B on a combined set of original and SRGAN-generated images.

### Step 4: Evaluation

Both classifiers are evaluated using the same held-out test identities. Metrics include accuracy, precision, recall, F1 and ROC-AUC; generated images are also evaluated with PSNR and SSIM against bicubic upscaling. Figures include image examples, losses, confusion matrices and ROC curves.

## 5. Results and Evaluation

The final notebook reports **13,288 images**, with **8,370 training**, **931 validation**, and **3,987 test** images (a 70% train allocation / 30% test split with validation taken from training). The final run reports **150 completed adversarial epochs**.

### Binary classification

| Metric | Model A (Original) | Model B (SRGAN) |
|---|---:|---:|
| Accuracy | **95.89%** | 92.78% |
| Precision | **97.53%** | 92.94% |
| Recall | **93.72%** | 91.76% |
| F1-score | **95.58%** | 92.35% |
| ROC-AUC | **0.9947** | 0.9839 |

### Super-resolution reconstruction

| Metric | Bicubic | SRGAN |
|---|---:|---:|
| PSNR (dB) | 25.1385 | **25.9379** |
| SSIM | 0.7630 | **0.7686** |

Model A exceeds Model B by **3.11 percentage points** in accuracy. SRGAN improves PSNR by **0.7994 dB** and SSIM by **0.0056** relative to bicubic on this evaluation.

### Figures

Upload these generated images from the experiment before expecting the links below to render:

- [Input scaling](images/image_transformations.png)
- [Model A training](images/classifier_A_training.png)
- [SRGAN training losses](images/srgan_training_losses.png)
- [SRGAN examples](images/srgan_generated_examples.png)
- [Model B training](images/classifier_B_training.png)
- [Classifier comparison](images/model_comparison.png)
- [Confusion matrices](images/confusion_matrices.png)
- [ROC curves](images/roc_comparison.png)

## 6. Conclusions and Limitations

The implemented SRGAN improves image reconstruction fidelity over bicubic interpolation, as measured by PSNR and SSIM. However, **Classifier A outperforms Classifier B**. Reconstruction improvements do not necessarily restore all discriminative information lost by downscaling. Discriminator loss was observed to approach zero during portions of training, which may indicate imbalance between the adversarial networks. Results are specific to this dataset and training setup. Future work could compare Model B with a classifier trained on bicubic-upscaled inputs and investigate more stable adversarial losses.

**Reproducibility notes:** The repository does not redistribute the full image dataset. The experiment's saved train/test CSV files may contain absolute paths specific to the original training machine; adapt or regenerate them consistently before rerunning the notebook. Large weights and full resume checkpoints should be linked in `models/README.md` if not uploaded to GitHub.

## 7. References

1. Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*, CVPR 2017. https://arxiv.org/abs/1609.04802
2. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. https://arxiv.org/abs/1512.03385
3. [PyTorch documentation](https://docs.pytorch.org/)
