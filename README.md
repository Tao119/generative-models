# Generative Models — Pure NumPy

Three generative models implemented from scratch with NumPy only (no deep learning frameworks).
Dataset: 2D Gaussian mixture, 4 clusters at corners of a unit square, 1000 samples.

## Models

| Model | File | Metric | Result | Runtime |
|-------|------|--------|--------|---------|
| VAE | `vae/vae.py` | Final ELBO loss | 2.47 | ~1.6 s |
| GAN (FC) | `gan/dcgan.py` | D_loss / G_loss at epoch 2000 | 0.45 / 1.38 | ~11.5 s |
| DDPM | `diffusion/ddpm.py` | Final MSE loss | 0.895 | ~1.5 s |

## Usage

```bash
python3 vae/vae.py         # → vae/vae_results.png
python3 gan/dcgan.py       # → gan/gan_results.png
python3 diffusion/ddpm.py  # → diffusion/ddpm_results.png, diffusion/ddpm_loss_curve.png
```

## Architectures

### VAE
- Encoder: 2 → 64 → 32 → (μ, log σ²) ∈ ℝ²
- Decoder: 2 → 32 → 64 → 2
- Loss: MSE reconstruction + β·KL, β=1

### GAN (fully-connected)
- Generator: z(8) → 32 → 64 → 2 (Tanh output)
- Discriminator: 2 → 64 → 32 → 1 (LeakyReLU + Sigmoid)
- BCE loss, Adam with β₁=0.5

### DDPM
- T=200 timesteps, linear β schedule [0.0001, 0.02]
- Noise network: (x_t ‖ t_embed) → 64 → 64 → 2, sinusoidal t embedding dim=16
- Loss: MSE on predicted vs actual noise

## Dependencies

```
numpy
matplotlib
```
