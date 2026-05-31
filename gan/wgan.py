"""
WGAN-GP — pure NumPy.

Same 2D 4-cluster Gaussian mixture data as dcgan.py.

Generator  : z(8) → Affine(8→32) → ReLU → Affine(32→64) → ReLU → Affine(64→2)
Critic      : x(2) → Affine(2→64) → LeakyReLU(0.2) → Affine(64→32)
              → LeakyReLU(0.2) → Affine(32→1)  [no Sigmoid]

WGAN-GP loss
  Critic loss  : E[D(fake)] - E[D(real)] + lambda_gp * E[(||grad_D(x̂)||₂ - 1)²]
  Generator loss: -E[D(fake)]
  lambda_gp = 10, critic updated 5× per generator step

Training: 3000 generator steps, batch=64
Comparison with DCGAN: W-distance logged every 500 steps.
Output: wgan_results.png  (real | DCGAN-final | WGAN-final, side by side)
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.layers import Affine, Relu, LeakyRelu, Sigmoid, Tanh, Adam, he_init


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_gaussian_mixture(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]], dtype=float)
    n_each = n // 4
    data = []
    for c in centers:
        data.append(rng.randn(n_each, 2) * 0.4 + c)
    return np.vstack(data)


# ---------------------------------------------------------------------------
# WGAN Generator (no Tanh)
# ---------------------------------------------------------------------------

class WGANGenerator:
    def __init__(self, z_dim=8):
        self.z_dim = z_dim
        self.fc1 = Affine(he_init(z_dim, 32), np.zeros(32))
        self.relu1 = Relu()
        self.fc2 = Affine(he_init(32, 64), np.zeros(64))
        self.relu2 = Relu()
        self.fc3 = Affine(he_init(64, 2), np.zeros(2))
        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3]

    def forward(self, z):
        h = z
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, dout):
        dh = dout
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        return dh

    def zero_grads(self):
        for layer in self.layers:
            for g in layer.grads:
                g[...] = 0.0

    def params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads
        return params, grads


# ---------------------------------------------------------------------------
# WGAN Critic (no Sigmoid)
# ---------------------------------------------------------------------------

class WGANCritic:
    def __init__(self):
        self.fc1 = Affine(he_init(2, 64), np.zeros(64))
        self.lrelu1 = LeakyRelu(0.2)
        self.fc2 = Affine(he_init(64, 32), np.zeros(32))
        self.lrelu2 = LeakyRelu(0.2)
        self.fc3 = Affine(he_init(32, 1), np.zeros(1))
        self.layers = [self.fc1, self.lrelu1, self.fc2, self.lrelu2, self.fc3]

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h  # (N, 1), unbounded

    def backward(self, dout):
        dh = dout
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        return dh

    def zero_grads(self):
        for layer in self.layers:
            for g in layer.grads:
                g[...] = 0.0

    def params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads
        return params, grads


# ---------------------------------------------------------------------------
# Gradient penalty (WGAN-GP)
# ---------------------------------------------------------------------------

def gradient_penalty(critic, real, fake, lambda_gp=10):
    """
    Compute gradient penalty: E[(||grad_D(x_hat)||_2 - 1)^2]
    where x_hat = eps*real + (1-eps)*fake, eps ~ U(0,1).

    We approximate the gradient numerically using finite differences
    to keep this pure-NumPy (no autograd).
    """
    N = real.shape[0]
    eps = np.random.uniform(0, 1, (N, 1))
    x_hat = eps * real + (1 - eps) * fake

    delta = 1e-4
    # Compute gradient via finite differences for each output dimension
    # x_hat: (N, 2) → D(x_hat): (N, 1)
    d_base = critic.forward(x_hat)  # (N, 1)

    grads = np.zeros_like(x_hat)  # (N, 2)
    for dim in range(x_hat.shape[1]):
        x_plus = x_hat.copy()
        x_plus[:, dim] += delta
        d_plus = critic.forward(x_plus)  # (N, 1)
        grads[:, dim] = ((d_plus - d_base) / delta).squeeze(axis=1)

    grad_norm = np.sqrt(np.sum(grads ** 2, axis=1) + 1e-8)  # (N,)
    penalty = np.mean((grad_norm - 1) ** 2)
    return penalty


# ---------------------------------------------------------------------------
# DCGAN reference (copied inline from dcgan.py logic, condensed)
# ---------------------------------------------------------------------------

class DCGANGenerator:
    def __init__(self, z_dim=8):
        self.z_dim = z_dim
        self.fc1 = Affine(he_init(z_dim, 32), np.zeros(32))
        self.relu1 = Relu()
        self.fc2 = Affine(he_init(32, 64), np.zeros(64))
        self.relu2 = Relu()
        self.fc3 = Affine(he_init(64, 2), np.zeros(2))
        self.tanh = Tanh()
        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3, self.tanh]

    def forward(self, z):
        h = z
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, dout):
        dh = dout
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        return dh

    def params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads
        return params, grads


class DCGANDiscriminator:
    def __init__(self):
        self.fc1 = Affine(he_init(2, 64), np.zeros(64))
        self.lrelu1 = LeakyRelu(0.2)
        self.fc2 = Affine(he_init(64, 32), np.zeros(32))
        self.lrelu2 = LeakyRelu(0.2)
        self.fc3 = Affine(he_init(32, 1), np.zeros(1))
        self.sigmoid = Sigmoid()
        self.layers = [self.fc1, self.lrelu1, self.fc2, self.lrelu2, self.fc3, self.sigmoid]

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, dout):
        dh = dout
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        return dh

    def zero_grads(self):
        for layer in self.layers:
            for g in layer.grads:
                g[...] = 0.0

    def params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads
        return params, grads


def bce_grad(pred, target):
    eps = 1e-7
    pred = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    grad = (pred - target) / (pred * (1 - pred) + eps) / pred.shape[0]
    return loss, grad


def train_dcgan(real_data, z_dim=8, n_steps=3000, batch_size=64, lr=2e-4):
    """Train DCGAN for n_steps generator steps (5 D steps per G step)."""
    N = real_data.shape[0]
    G = DCGANGenerator(z_dim)
    D = DCGANDiscriminator()
    opt_G = Adam(lr=lr, beta1=0.5)
    opt_D = Adam(lr=lr, beta1=0.5)

    for step in range(1, n_steps + 1):
        for _ in range(5):
            idx = np.random.randint(0, N, batch_size)
            xb = real_data[idx]
            z = np.random.randn(batch_size, z_dim)
            fake = G.forward(z)

            combined = np.concatenate([xb, fake], axis=0)
            labels = np.concatenate([np.ones((batch_size, 1)),
                                     np.zeros((batch_size, 1))], axis=0)
            D.zero_grads()
            d_pred = D.forward(combined)
            d_loss, d_grad = bce_grad(d_pred, labels)
            D.backward(d_grad)
            d_p, d_g = D.params_and_grads()
            opt_D.update(d_p, d_g)

        # G step
        z2 = np.random.randn(batch_size, z_dim)
        fake2 = G.forward(z2)
        D.zero_grads()
        d_fake = D.forward(fake2)
        g_loss, g_grad = bce_grad(d_fake, np.ones((batch_size, 1)))
        d_inp_g = D.backward(g_grad)
        G.backward(d_inp_g)
        g_p, g_g = G.params_and_grads()
        opt_G.update(g_p, g_g)

    return G


# ---------------------------------------------------------------------------
# WGAN-GP training
# ---------------------------------------------------------------------------

def train_wgan_gp(real_data, z_dim=8, n_steps=3000, batch_size=64,
                  lr=1e-4, lambda_gp=10, n_critic=5):
    N = real_data.shape[0]
    G = WGANGenerator(z_dim)
    C = WGANCritic()
    opt_G = Adam(lr=lr, beta1=0.0, beta2=0.9)
    opt_C = Adam(lr=lr, beta1=0.0, beta2=0.9)

    w_distances = []

    for step in range(1, n_steps + 1):
        # ----- Critic steps -----
        for _ in range(n_critic):
            idx = np.random.randint(0, N, batch_size)
            real = real_data[idx]
            z = np.random.randn(batch_size, z_dim)
            fake = G.forward(z)

            C.zero_grads()
            d_real = C.forward(real)   # (N, 1)
            d_fake = C.forward(fake)   # (N, 1)

            # Wasserstein critic loss: E[D(fake)] - E[D(real)]
            w_loss = np.mean(d_fake) - np.mean(d_real)
            gp = gradient_penalty(C, real, fake, lambda_gp)
            c_loss = w_loss + lambda_gp * gp

            # Manual gradient:  d(E[D(fake)])/d(params) comes from backward on fake
            #                   d(-E[D(real)])/d(params) comes from backward on real
            # We need combined backward pass manually.
            # Gradient for fake part: +1/N for each fake sample
            d_grad_fake = np.ones((batch_size, 1)) / batch_size
            # Gradient for real part: -1/N for each real sample
            d_grad_real = -np.ones((batch_size, 1)) / batch_size

            # Re-forward real and fake so layer caches are fresh
            C.zero_grads()
            d_real2 = C.forward(real)
            C.backward(d_grad_real)
            # Accumulate gradients from fake
            d_fake2 = C.forward(fake)
            C.backward(d_grad_fake)

            c_p, c_g = C.params_and_grads()
            opt_C.update(c_p, c_g)

        # ----- Generator step -----
        G.zero_grads()
        z3 = np.random.randn(batch_size, z_dim)
        fake3 = G.forward(z3)
        C.zero_grads()
        d_fake3 = C.forward(fake3)   # (N, 1)
        # Generator loss: -E[D(fake)]
        # d(-E[D(fake)])/d(D(fake)) = -1/N per sample
        d_grad_g = -np.ones((batch_size, 1)) / batch_size
        d_inp = C.backward(d_grad_g)   # gradient w.r.t. fake3
        G.backward(d_inp)
        g_p, g_g = G.params_and_grads()
        opt_G.update(g_p, g_g)

        # Log W-distance estimate
        if step % 500 == 0:
            idx_eval = np.random.randint(0, N, 256)
            real_eval = real_data[idx_eval]
            z_eval = np.random.randn(256, z_dim)
            fake_eval = G.forward(z_eval)
            d_r = C.forward(real_eval)
            d_f = C.forward(fake_eval)
            w_dist = float(np.mean(d_r) - np.mean(d_f))
            w_distances.append((step, w_dist))
            print(f"  Step {step:4d}  W-dist ≈ {w_dist:.4f}")

    return G, w_distances


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(0)
    t0 = time.time()

    real_data = make_gaussian_mixture(n=1000)
    # Normalize for training
    std = real_data.std(axis=0)
    real_norm = real_data / std

    print("=== Training DCGAN (reference) ===")
    G_dcgan = train_dcgan(real_norm, z_dim=8, n_steps=3000, batch_size=64, lr=2e-4)
    z_final = np.random.randn(500, 8)
    dcgan_samples = G_dcgan.forward(z_final) * std

    print("\n=== Training WGAN-GP ===")
    G_wgan, w_distances = train_wgan_gp(
        real_norm, z_dim=8, n_steps=3000, batch_size=64,
        lr=1e-4, lambda_gp=10, n_critic=5
    )
    z_final2 = np.random.randn(500, 8)
    wgan_samples = G_wgan.forward(z_final2) * std

    elapsed = time.time() - t0
    print(f"\nRuntime: {elapsed:.1f}s")

    print("\n=== W-distance at intervals (WGAN-GP) ===")
    for step, wd in w_distances:
        print(f"  Step {step}: {wd:.4f}")

    # Distribution quality: compare std of generated vs real
    print("\n=== Distribution quality (std comparison) ===")
    print(f"  Real std:   {real_data.std(axis=0)}")
    print(f"  DCGAN std:  {dcgan_samples.std(axis=0)}")
    print(f"  WGAN-GP std:{wgan_samples.std(axis=0)}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xlim, ylim = (-5, 5), (-5, 5)

    axes[0].scatter(real_data[:, 0], real_data[:, 1], s=8, alpha=0.4,
                    label='real', color='tab:blue')
    axes[0].set_title('Real data')
    axes[0].set_xlim(*xlim); axes[0].set_ylim(*ylim)
    axes[0].set_aspect('equal')

    axes[1].scatter(real_data[:, 0], real_data[:, 1], s=8, alpha=0.2, color='tab:blue')
    axes[1].scatter(dcgan_samples[:, 0], dcgan_samples[:, 1], s=8, alpha=0.5,
                    color='tab:red', label='DCGAN')
    axes[1].set_title('DCGAN (3000 G-steps)')
    axes[1].set_xlim(*xlim); axes[1].set_ylim(*ylim)
    axes[1].set_aspect('equal')
    axes[1].legend(markerscale=2, fontsize=8)

    axes[2].scatter(real_data[:, 0], real_data[:, 1], s=8, alpha=0.2, color='tab:blue')
    axes[2].scatter(wgan_samples[:, 0], wgan_samples[:, 1], s=8, alpha=0.5,
                    color='tab:green', label='WGAN-GP')
    axes[2].set_title('WGAN-GP (3000 G-steps)')
    axes[2].set_xlim(*xlim); axes[2].set_ylim(*ylim)
    axes[2].set_aspect('equal')
    axes[2].legend(markerscale=2, fontsize=8)

    plt.suptitle('GAN Comparison: Real vs DCGAN vs WGAN-GP', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wgan_results.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
