# Cats vs. Dogs Dataset

This project uses the Cats vs. Dogs image dataset for binary image classification and super-resolution.

## Dataset Preparation

Images are organized into two classes:

- Cat: Label 0
- Dog: Label 1

The final experiment used 13,288 images, divided into:

| Partition | Images |
|---|---:|
| Training | 8,370 |
| Validation | 931 |
| Testing | 3,987 |

The dataset is used in two preprocessing pipelines:

- Classifier A: Original images resized to 128 × 128.
- SRGAN: Images downscaled to 32 × 32 and reconstructed to 128 × 128.
- Classifier B: SRGAN-generated 128 × 128 images.

Download the original Cats vs. Dogs dataset from the dataset provider and update the dataset path in the notebook before running it.

The full image dataset is not included in the GitHub repository because of its size.
