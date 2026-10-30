# Trained model checkpoints

The user-generated model files are **not included** in the starter repository and must be copied from the completed experiment. Example final files from `SRGAN_Midterm/checkpoints/`:

- `classifier_A_latest_best.pth` — best Model A weights
- `generator_final.pth` — SRGAN generator weights
- `discriminator_final.pth` — SRGAN discriminator weights
- `classifier_B_latest_best.pth` — best Model B weights
- `srgan_latest.pth` — **large resume checkpoint**, containing model/optimizer/training states

Recommended: upload final weights to GitHub Releases or an accessible Drive folder and replace the following placeholders with real links:

- Model A: **ADD VERIFIED LINK**
- Generator: **ADD VERIFIED LINK**
- Discriminator: **ADD VERIFIED LINK**
- Model B: **ADD VERIFIED LINK**
- SRGAN resume checkpoint (optional): **ADD VERIFIED LINK**

Do not commit hundreds of periodic checkpoints. `*.pth` files are ignored by `.gitignore`; use Git LFS deliberately if you want them tracked in Git instead.
