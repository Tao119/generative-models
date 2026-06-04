"""
ddpm_improved.py — Improved DDPM with Classifier-Free Guidance and DDIM Sampling

Features:
- Linear and Cosine noise schedules
- U-Net style denoising network with skip connections
- Classifier-Free Guidance (CFG)
- DDIM sampling (10 steps instead of 1000)
- 2D 8-Gaussian labeled dataset
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.layers import Affine, Relu, Adam, he_init


# ─────────────────────────────────────────────────────────────
# Dataset: 8 Gaussian classes (labels 0-7)
# ─────────────────────────────────────────────────────────────

def make_8gaussian_mixture(n=2000, seed=42):
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    radius = 2.5
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    n_each = n // 8
    data, labels = [], []
    for k, c in enumerate(centers):
        data.append(rng.randn(n_each, 2) * 0.3 + c)
        labels.append(np.full(n_each, k, dtype=int))
    return np.vstack(data), np.concatenate(labels)


# ─────────────────────────────────────────────────────────────
# Noise Schedules
# ─────────────────────────────────────────────────────────────

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    beta = np.linspace(beta_start, beta_end, T)
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    return beta, alpha, alpha_bar


def cosine_schedule(T, s=0.008):
    t = np.arange(T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    alpha_bar = np.clip(alpha_bar, 1e-5, 1.0)
    alpha_bar_prev = np.concatenate([[1.0], alpha_bar[:-1]])
    alpha = alpha_bar / alpha_bar_prev
    alpha = np.clip(alpha, 1e-5, 1.0)
    beta = 1.0 - alpha
    return beta[1:], alpha[1:], alpha_bar[1:]


# ─────────────────────────────────────────────────────────────
# Sinusoidal Embeddings
# ─────────────────────────────────────────────────────────────

def sinusoidal_embedding(t, dim=32):
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)


def class_embedding(c, num_classes=8, dim=16):
    """One-hot → linear projection embedding."""
    onehot = np.zeros((len(c), num_classes), dtype=float)
    onehot[np.arange(len(c)), c] = 1.0
    return onehot


# ─────────────────────────────────────────────────────────────
# U-Net style network with skip connections (MLP-based for 2D)
# ─────────────────────────────────────────────────────────────

class UNetDenoisingNet:
    """
    MLP U-Net analogue for 2D data.
    Encoder: 2 → 128 → 64
    Bottleneck: 64 → 64
    Decoder: 64 + skip(128) → 128 → 2
    Time and class conditioning injected at each block.
    """

    def __init__(self, t_dim=32, c_dim=8, hidden=128, null_class_id=8):
        # Input dim: data(2) + t_emb(t_dim) + c_emb(c_dim)
        cond_dim = t_dim + c_dim
        inp_dim = 2 + cond_dim

        # Encoder block 1: inp → hidden
        self.enc1_w = Affine(he_init(inp_dim, hidden), np.zeros(hidden))
        self.enc1_r = Relu()

        # Encoder block 2: hidden + cond → hidden//2
        self.enc2_w = Affine(he_init(hidden + cond_dim, hidden // 2), np.zeros(hidden // 2))
        self.enc2_r = Relu()

        # Bottleneck: hidden//2 + cond → hidden//2
        self.bot_w = Affine(he_init(hidden // 2 + cond_dim, hidden // 2), np.zeros(hidden // 2))
        self.bot_r = Relu()

        # Decoder block 1: hidden//2 + skip(hidden) + cond → hidden
        self.dec1_w = Affine(he_init(hidden // 2 + hidden + cond_dim, hidden), np.zeros(hidden))
        self.dec1_r = Relu()

        # Decoder block 2: hidden + cond → 2
        self.dec2_w = Affine(he_init(hidden + cond_dim, 2), np.zeros(2))

        self.t_dim = t_dim
        self.c_dim = c_dim
        self.null_class_id = null_class_id
        self.num_classes = null_class_id  # 8 real classes

        # Null class embedding (for unconditional branch)
        self.null_emb = np.zeros(c_dim)

        # All layers for param collection
        self._all_layers = [
            self.enc1_w, self.enc2_w, self.bot_w, self.dec1_w, self.dec2_w
        ]

        # Cache for backward
        self._cache = {}

    def _get_c_emb(self, c):
        """c: int array, null_class_id → null embedding."""
        emb = np.zeros((len(c), self.c_dim))
        mask = c < self.null_class_id
        emb[mask] = class_embedding(c[mask], self.num_classes, self.c_dim)
        return emb

    def forward(self, xt, t_emb, c_emb):
        cond = np.concatenate([t_emb, c_emb], axis=1)

        # Encoder block 1
        h0 = np.concatenate([xt, cond], axis=1)
        h1 = self.enc1_r.forward(self.enc1_w.forward(h0))

        # Encoder block 2
        h1c = np.concatenate([h1, cond], axis=1)
        h2 = self.enc2_r.forward(self.enc2_w.forward(h1c))

        # Bottleneck
        h2c = np.concatenate([h2, cond], axis=1)
        hb = self.bot_r.forward(self.bot_w.forward(h2c))

        # Decoder block 1 (with skip from h1)
        hb_skip = np.concatenate([hb, h1, cond], axis=1)
        hd1 = self.dec1_r.forward(self.dec1_w.forward(hb_skip))

        # Decoder block 2
        hd1c = np.concatenate([hd1, cond], axis=1)
        out = self.dec2_w.forward(hd1c)

        self._cache = {
            'xt': xt, 't_emb': t_emb, 'c_emb': c_emb, 'cond': cond,
            'h0': h0, 'h1': h1, 'h1c': h1c,
            'h2': h2, 'h2c': h2c, 'hb': hb,
            'hb_skip': hb_skip, 'hd1': hd1, 'hd1c': hd1c
        }
        return out

    def backward(self, dout):
        c = self._cache
        # dec2
        dhd1c = self.dec2_w.backward(dout)
        dhd1 = dhd1c[:, :c['hd1'].shape[1]]

        # dec1_relu + dec1
        dhd1 = self.dec1_r.backward(dhd1)
        dhb_skip = self.dec1_w.backward(dhd1)
        dhb = dhb_skip[:, :c['hb'].shape[1]]

        # bottleneck
        dhb = self.bot_r.backward(dhb)
        dh2c = self.bot_w.backward(dhb)
        dh2 = dh2c[:, :c['h2'].shape[1]]

        # enc2
        dh2 = self.enc2_r.backward(dh2)
        dh1c = self.enc2_w.backward(dh2)
        dh1 = dh1c[:, :c['h1'].shape[1]]

        # enc1
        dh1 = self.enc1_r.backward(dh1)
        dh0 = self.enc1_w.backward(dh1)

        return dh0[:, :2]  # gradient w.r.t. xt

    def params_and_grads(self):
        params, grads = [], []
        for layer in self._all_layers:
            params += layer.params
            grads += layer.grads
        return params, grads


# ─────────────────────────────────────────────────────────────
# Improved DDPM with CFG
# ─────────────────────────────────────────────────────────────

class ImprovedDDPM:
    def __init__(self, T=300, schedule='cosine', t_dim=32, c_dim=8,
                 lr=1e-3, num_classes=8, p_uncond=0.15):
        self.T = T
        self.t_dim = t_dim
        self.num_classes = num_classes
        self.p_uncond = p_uncond
        self.null_class_id = num_classes  # special token for unconditional

        if schedule == 'linear':
            self.beta, self.alpha, self.alpha_bar = linear_schedule(T)
        else:
            self.beta, self.alpha, self.alpha_bar = cosine_schedule(T)

        self.net = UNetDenoisingNet(t_dim=t_dim, c_dim=c_dim, hidden=128,
                                    null_class_id=self.null_class_id)
        self.optimizer = Adam(lr=lr)

    def q_sample(self, x0, t):
        ab = self.alpha_bar[t][:, None]
        eps = np.random.randn(*x0.shape)
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        return xt, eps

    def _cfg_forward(self, xt, t_arr, c_arr):
        """Forward pass with classifier-free guidance conditioning."""
        t_emb = sinusoidal_embedding(t_arr.astype(float), dim=self.t_dim)
        c_emb = self.net._get_c_emb(c_arr)
        return self.net.forward(xt, t_emb, c_emb)

    def loss_and_backward(self, x0, t, c):
        xt, eps = self.q_sample(x0, t)

        # Randomly drop class labels (CFG training)
        mask = np.random.rand(len(c)) < self.p_uncond
        c_train = c.copy()
        c_train[mask] = self.null_class_id

        t_emb = sinusoidal_embedding(t.astype(float), dim=self.t_dim)
        c_emb = self.net._get_c_emb(c_train)
        eps_pred = self.net.forward(xt, t_emb, c_emb)

        N = x0.shape[0]
        diff = eps_pred - eps
        loss = np.sum(diff ** 2) / N
        dout = 2 * diff / N
        self.net.backward(dout)
        return loss

    def step(self, x0, t, c):
        loss = self.loss_and_backward(x0, t, c)
        params, grads = self.net.params_and_grads()
        self.optimizer.update(params, grads)
        return loss

    # ── DDIM sampling (fast inference) ──────────────────────────

    def ddim_sample(self, n, target_class=None, w=3.0, ddim_steps=10):
        """
        DDIM sampling with Classifier-Free Guidance.
        w: guidance scale (0 = unconditional, >0 = guided)
        target_class: None = random classes, int = fixed class
        """
        x = np.random.randn(n, 2)

        # Select timestep subset (evenly spaced)
        step_indices = np.linspace(0, self.T - 1, ddim_steps + 1, dtype=int)
        step_indices = step_indices[::-1]  # T → 0

        # Set class labels
        if target_class is None:
            c_arr = np.random.randint(0, self.num_classes, size=n)
        else:
            c_arr = np.full(n, target_class, dtype=int)
        c_null = np.full(n, self.null_class_id, dtype=int)

        for i in range(len(step_indices) - 1):
            t_cur = step_indices[i]
            t_next = step_indices[i + 1]

            t_arr = np.full(n, t_cur, dtype=int)

            # Conditional prediction
            eps_cond = self._cfg_forward(x, t_arr, c_arr)
            # Unconditional prediction
            eps_uncond = self._cfg_forward(x, t_arr, c_null)

            # CFG mixing
            eps_hat = eps_uncond + w * (eps_cond - eps_uncond)

            ab_cur = self.alpha_bar[t_cur]
            ab_next = self.alpha_bar[t_next] if t_next >= 0 else 1.0

            # DDIM update (deterministic)
            x0_pred = (x - np.sqrt(1 - ab_cur) * eps_hat) / np.sqrt(ab_cur)
            x0_pred = np.clip(x0_pred, -5, 5)

            x = np.sqrt(ab_next) * x0_pred + np.sqrt(1 - ab_next) * eps_hat

        return x, c_arr

    # ── Standard DDPM sampling ───────────────────────────────────

    def ddpm_sample(self, n, target_class=None, w=3.0):
        x = np.random.randn(n, 2)
        if target_class is None:
            c_arr = np.random.randint(0, self.num_classes, size=n)
        else:
            c_arr = np.full(n, target_class, dtype=int)
        c_null = np.full(n, self.null_class_id, dtype=int)

        for t_idx in reversed(range(self.T)):
            t_arr = np.full(n, t_idx, dtype=int)
            eps_cond = self._cfg_forward(x, t_arr, c_arr)
            eps_uncond = self._cfg_forward(x, t_arr, c_null)
            eps_hat = eps_uncond + w * (eps_cond - eps_uncond)

            ab = self.alpha_bar[t_idx]
            a = self.alpha[t_idx]
            b = self.beta[t_idx]

            x0_pred = (x - np.sqrt(1 - ab) * eps_hat) / np.sqrt(ab)
            x0_pred = np.clip(x0_pred, -5, 5)

            mean = (np.sqrt(a) * (1 - ab / a) * x
                    + np.sqrt(ab / a) * (1 - a) * x0_pred) / (1 - ab + 1e-10)
            if t_idx > 0:
                x = mean + np.sqrt(b) * np.random.randn(*x.shape)
            else:
                x = mean

        return x, c_arr


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(model, data, labels, epochs=500, batch_size=256):
    N = data.shape[0]
    losses = []
    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        total, n_b = 0.0, 0
        for i in range(0, N - batch_size + 1, batch_size):
            xb = data[idx[i:i + batch_size]]
            lb = labels[idx[i:i + batch_size]]
            t = np.random.randint(0, model.T, size=xb.shape[0])
            loss = model.step(xb, t, lb)
            total += loss
            n_b += 1
        losses.append(total / max(n_b, 1))
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={losses[-1]:.6f}")
    return losses


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(0)
    t0 = time.time()

    # ── Dataset ──
    raw_data, labels = make_8gaussian_mixture(n=2000)
    std = raw_data.std(axis=0) + 1e-8
    data = raw_data / std

    print("=== Improved DDPM with CFG + DDIM ===")
    print(f"Data shape: {data.shape}, Classes: {np.unique(labels)}")
    print(f"Schedule: cosine,  T=300")

    # ── Train ──
    model = ImprovedDDPM(T=300, schedule='cosine', t_dim=32, c_dim=8, lr=1e-3)
    print("\nTraining...")
    losses = train(model, data, labels, epochs=500, batch_size=256)

    elapsed = time.time() - t0
    print(f"\nTraining done. Final loss: {losses[-1]:.6f}  ({elapsed:.1f}s)")

    # ── DDIM vs DDPM sampling ──
    print("\nDDIM sampling (10 steps)...")
    t1 = time.time()
    ddim_samples, ddim_classes = model.ddim_sample(800, w=3.0)
    ddim_time = time.time() - t1
    ddim_samples_unnorm = ddim_samples * std

    print(f"DDPM sampling (full {model.T} steps)...")
    t2 = time.time()
    ddpm_samples, ddpm_classes = model.ddpm_sample(200, w=3.0)
    ddpm_time = time.time() - t2
    ddpm_samples_unnorm = ddpm_samples * std

    print(f"DDIM: {ddim_time:.2f}s ({model.T}→10 steps) | DDPM: {ddpm_time:.2f}s ({model.T} steps)")

    # ── Class-conditional grid ──
    class_colors = plt.cm.tab10(np.linspace(0, 1, 8))

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Panel 0: Real data
    ax = axes[0]
    for k in range(8):
        m = labels == k
        ax.scatter(raw_data[m, 0], raw_data[m, 1], s=8, alpha=0.5, color=class_colors[k])
    ax.set_title('Real Data (8 classes)', fontsize=11, fontweight='bold')
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')

    # Panel 1: DDIM random
    ax = axes[1]
    for k in range(8):
        m = ddim_classes == k
        if m.any():
            ax.scatter(ddim_samples_unnorm[m, 0], ddim_samples_unnorm[m, 1],
                       s=8, alpha=0.5, color=class_colors[k])
    ax.set_title(f'DDIM (10 steps, w=3)\n{ddim_time:.2f}s', fontsize=10)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')

    # Panel 2: DDPM random
    ax = axes[2]
    for k in range(8):
        m = ddpm_classes == k
        if m.any():
            ax.scatter(ddpm_samples_unnorm[m, 0], ddpm_samples_unnorm[m, 1],
                       s=8, alpha=0.5, color=class_colors[k])
    ax.set_title(f'DDPM ({model.T} steps, w=3)\n{ddpm_time:.2f}s', fontsize=10)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')

    # Panel 3: Loss curve
    ax = axes[3]
    ax.plot(losses, color='steelblue')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss (CFG)')
    ax.set_yscale('log')

    # Panels 4-11: Class-conditional DDIM (classes 0-7)
    for k in range(8):
        samp, _ = model.ddim_sample(200, target_class=k, w=3.0)
        samp_unnorm = samp * std
        ax = axes[4 + k]
        ax.scatter(raw_data[labels == k, 0], raw_data[labels == k, 1],
                   s=8, alpha=0.3, color='gray', label='real')
        ax.scatter(samp_unnorm[:, 0], samp_unnorm[:, 1],
                   s=8, alpha=0.6, color=class_colors[k], label=f'gen cls {k}')
        ax.set_title(f'Class {k} conditional (DDIM)', fontsize=9)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')
        ax.legend(markerscale=2, fontsize=7)

    plt.suptitle('Improved DDPM: Cosine Schedule + CFG + DDIM', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'ddpm_improved_results.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved {out_path}")

    # ── Schedule comparison ──
    t_arr = np.arange(300)
    _, _, ab_lin = linear_schedule(300)
    _, _, ab_cos = cosine_schedule(300)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(t_arr, ab_lin, label='Linear', color='steelblue')
    ax2.plot(t_arr, ab_cos, label='Cosine', color='darkorange')
    ax2.set_xlabel('Timestep t'); ax2.set_ylabel('ᾱₜ (signal retention)')
    ax2.set_title('Noise Schedules: Linear vs Cosine')
    ax2.legend()
    plt.tight_layout()
    sched_path = os.path.join(os.path.dirname(__file__), 'ddpm_schedules.png')
    plt.savefig(sched_path, dpi=120)
    plt.close()
    print(f"Saved {sched_path}")


if __name__ == '__main__':
    main()
