# Trained Models

This directory documents the trained PyTorch models for the SRGAN Cats vs. Dogs binary classification project.

## Available Models

| Model | Checkpoint | Description |
|---|---|---|
| Classifier A | `classifier_A_latest_best.pth` | ResNet18 trained on original 128 × 128 images |
| SRGAN Generator | `generator_final.pth` | Generates 128 × 128 images from 32 × 32 images |
| SRGAN Discriminator | `discriminator_final.pth` | Distinguishes real and generated high-resolution images |
| Classifier B | `classifier_B_latest_best.pth` | ResNet18 trained on SRGAN-generated images |

## Download Checkpoints

The trained model weights are available through the project's GitHub Releases.

[Download trained models](https://github.com/Radhika-95/SARGAN/releases)

## Training Configuration

- Framework: PyTorch
- Classification architecture: ResNet18
- SRGAN generator: 8 residual blocks
- Upscaling factor: 4×
- SRGAN adversarial training: 150 epochs
- Generator input: 32 × 32 RGB
- Generator output: 128 × 128 RGB

The Jupyter Notebook contains the model architectures, training procedure, checkpointing, and evaluation code.
