"""
vqvae.py — Vector Quantized VAE (VQ-VAE) in pure NumPy

Architecture:
  Encoder:   MLP  input(2) → 32 → 16 → codebook_dim(8)
  Codebook:  K=16 discrete codes, each dim=8
  Decoder:   MLP  8 → 16 → 32 → input(2)

Training:
  Loss = reconstruction_loss + commitment_loss + codebook_loss
  Straight-through estimator for backward pass

Dataset: 2D 8-Gaussian mixture (labels 0-7)
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
# Dataset
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
# MLP Encoder
# ─────────────────────────────────────────────────────────────

class Encoder:
    """Input(2) → 32 → 16 → codebook_dim(8)"""

    def __init__(self, in_dim=2, hidden1=32, hidden2=16, out_dim=8):
        self.fc1 = Affine(he_init(in_dim, hidden1), np.zeros(hidden1))
        self.r1 = Relu()
        self.fc2 = Affine(he_init(hidden1, hidden2), np.zeros(hidden2))
        self.r2 = Relu()
        self.fc3 = Affine(he_init(hidden2, out_dim), np.zeros(out_dim))
        self.layers = [self.fc1, self.r1, self.fc2, self.r2, self.fc3]

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

    def params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads
        return params, grads


# ─────────────────────────────────────────────────────────────
# MLP Decoder
# ─────────────────────────────────────────────────────────────

class Decoder:
    """codebook_dim(8) → 16 → 32 → input(2)"""

    def __init__(self, in_dim=8, hidden1=16, hidden2=32, out_dim=2):
        self.fc1 = Affine(he_init(in_dim, hidden1), np.zeros(hidden1))
        self.r1 = Relu()
        self.fc2 = Affine(he_init(hidden1, hidden2), np.zeros(hidden2))
        self.r2 = Relu()
        self.fc3 = Affine(he_init(hidden2, out_dim), np.zeros(out_dim))
        self.layers = [self.fc1, self.r1, self.fc2, self.r2, self.fc3]

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


# ─────────────────────────────────────────────────────────────
# VQ-VAE
# ─────────────────────────────────────────────────────────────

class VQVAE:
    """
    Pure NumPy VQ-VAE.

    forward(x) returns (x_recon, z_e, z_q, indices)
    Losses:
      L_recon = MSE(x_recon, x)
      L_commit = beta * ||sg(z_q) - z_e||^2   (commitment loss)
      L_codebook = ||z_q - sg(z_e)||^2
    Straight-through: gradients flow through quantisation step unchanged.
    """

    def __init__(self, K=16, D=8, beta=0.25, lr=1e-3):
        self.K = K       # number of codebook entries
        self.D = D       # codebook/embedding dimension
        self.beta = beta

        self.encoder = Encoder(in_dim=2, hidden1=32, hidden2=16, out_dim=D)
        self.decoder = Decoder(in_dim=D, hidden1=16, hidden2=32, out_dim=2)

        # Codebook: K × D, initialised small
        self.codebook = np.random.randn(K, D) * 0.1

        self.enc_opt = Adam(lr=lr)
        self.dec_opt = Adam(lr=lr)

        # EMA for codebook update (more stable than gradient)
        self._ema_count = np.ones(K)
        self._ema_sum = self.codebook.copy()
        self._ema_decay = 0.99

        # Cache for backward
        self._cache = {}

    def quantize(self, z_e):
        """
        z_e: (N, D)
        Returns z_q (N, D) and indices (N,).
        """
        # L2 distances: ||z_e||^2 + ||e_k||^2 - 2 z_e·e_k^T
        dist = (
            np.sum(z_e ** 2, axis=1, keepdims=True)
            + np.sum(self.codebook ** 2, axis=1)
            - 2 * z_e @ self.codebook.T
        )  # (N, K)
        indices = np.argmin(dist, axis=1)  # (N,)
        z_q = self.codebook[indices]       # (N, D)
        return z_q, indices

    def forward(self, x):
        z_e = self.encoder.forward(x)
        z_q, indices = self.quantize(z_e)

        # Straight-through: pass z_q but gradient = grad of z_e
        # z_q_st = z_e + (z_q - z_e) = z_q, but backward sees identity on z_e
        z_q_st = z_e + (z_q - z_e)  # same value, different graph

        x_recon = self.decoder.forward(z_q_st)

        self._cache = {'x': x, 'z_e': z_e, 'z_q': z_q, 'indices': indices, 'z_q_st': z_q_st}
        return x_recon, z_e, z_q, indices

    def loss(self, x, x_recon, z_e, z_q):
        """Compute VQ-VAE loss components."""
        N = x.shape[0]
        L_recon = np.sum((x_recon - x) ** 2) / N
        L_codebook = np.sum((z_q - z_e.copy()) ** 2) / N   # codebook moves to z_e
        L_commit = self.beta * np.sum((z_e - z_q.copy()) ** 2) / N  # z_e moves to z_q
        total = L_recon + L_codebook + L_commit
        return total, L_recon, L_codebook, L_commit

    def backward_and_update(self, x):
        """Full forward-backward-update step."""
        x_recon, z_e, z_q, indices = self.forward(x)

        total, L_recon, L_codebook, L_commit = self.loss(x, x_recon, z_e, z_q)

        N = x.shape[0]

        # ── Decoder backward ──
        # dL/dx_recon = 2/N * (x_recon - x)
        d_recon = 2.0 / N * (x_recon - x)
        d_z_q_st = self.decoder.backward(d_recon)

        # Straight-through: d_z_e = d_z_q_st (gradient passes through quantisation)
        d_z_e = d_z_q_st.copy()

        # ── Commitment loss gradient for encoder ──
        # L_commit = beta * ||z_e - sg(z_q)||^2
        d_z_e += 2.0 * self.beta / N * (z_e - z_q)

        # ── Encoder backward ──
        self.encoder.backward(d_z_e)

        # ── EMA Codebook update ──
        # More stable than gradient update
        for k in range(self.K):
            mask = indices == k
            n_k = mask.sum()
            if n_k > 0:
                self._ema_count[k] = (self._ema_decay * self._ema_count[k]
                                       + (1 - self._ema_decay) * n_k)
                self._ema_sum[k] = (self._ema_decay * self._ema_sum[k]
                                     + (1 - self._ema_decay) * z_e[mask].sum(axis=0))
                self.codebook[k] = self._ema_sum[k] / (self._ema_count[k] + 1e-8)

        # ── Update encoder and decoder params ──
        enc_p, enc_g = self.encoder.params_and_grads()
        dec_p, dec_g = self.decoder.params_and_grads()
        self.enc_opt.update(enc_p, enc_g)
        self.dec_opt.update(dec_p, dec_g)

        return total, L_recon, L_codebook, L_commit, indices


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(model, data, epochs=600, batch_size=256):
    N = data.shape[0]
    history = {'total': [], 'recon': [], 'codebook': [], 'commit': []}

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        totals = []
        for i in range(0, N - batch_size + 1, batch_size):
            xb = data[idx[i:i + batch_size]]
            total, L_r, L_c, L_cm, _ = model.backward_and_update(xb)
            totals.append(total)

        history['total'].append(np.mean(totals))
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"total={history['total'][-1]:.5f}")

    return history


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    t0 = time.time()

    # ── Dataset ──
    raw_data, labels = make_8gaussian_mixture(n=2000)
    std = raw_data.std(axis=0) + 1e-8
    data = raw_data / std

    print("=== VQ-VAE (Pure NumPy) ===")
    print(f"Data shape: {data.shape}, Codebook K=16, D=8")

    # ── Train ──
    model = VQVAE(K=16, D=8, beta=0.25, lr=2e-3)
    print("\nTraining...")
    history = train(model, data, epochs=600, batch_size=256)

    elapsed = time.time() - t0
    print(f"\nTraining done. Final total loss: {history['total'][-1]:.6f}  ({elapsed:.1f}s)")

    # ── Evaluate code usage ──
    x_recon, z_e, z_q, indices = model.forward(data)
    code_counts = np.bincount(indices, minlength=model.K)
    active_codes = (code_counts > 0).sum()
    print(f"\nCodebook usage: {active_codes}/{model.K} active codes")
    for k in range(model.K):
        bar = '█' * int(code_counts[k] / max(code_counts) * 20)
        print(f"  Code {k:2d}: {code_counts[k]:4d} | {bar}")

    # ── Reconstruction quality ──
    recon_unnorm = x_recon * std
    recon_mse = np.mean((recon_unnorm - raw_data) ** 2)
    print(f"\nReconstruction MSE (unnorm): {recon_mse:.6f}")

    # ── Code assignment per cluster ──
    code_per_cluster = {}
    for k in range(8):
        m = labels == k
        if m.any():
            idxs = indices[m]
            unique, cnts = np.unique(idxs, return_counts=True)
            top = unique[np.argmax(cnts)]
            code_per_cluster[k] = (top, cnts.max(), idxs.shape[0])
    print("\nDominant code per cluster:")
    for k, (code, cnt, total_k) in code_per_cluster.items():
        print(f"  Cluster {k}: code {code:2d}  ({cnt}/{total_k}  {100*cnt/total_k:.0f}%)")

    # ── Visualisation ──
    class_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    code_colors = plt.cm.Set3(np.linspace(0, 1, model.K))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: Real data coloured by class
    ax = axes[0, 0]
    for k in range(8):
        m = labels == k
        ax.scatter(raw_data[m, 0], raw_data[m, 1], s=10, alpha=0.6,
                   color=class_colors[k], label=f'cls {k}')
    ax.set_title('Real Data (by class)', fontsize=11)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')
    ax.legend(markerscale=1.5, fontsize=7, ncol=2)

    # Panel 2: Data coloured by assigned code
    ax = axes[0, 1]
    for k in range(model.K):
        m = indices == k
        if m.any():
            ax.scatter(raw_data[m, 0], raw_data[m, 1], s=10, alpha=0.6,
                       color=code_colors[k], label=f'code {k}')
    ax.set_title('Real Data (by VQ code)', fontsize=11)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')

    # Panel 3: Reconstruction
    ax = axes[0, 2]
    ax.scatter(raw_data[:, 0], raw_data[:, 1], s=6, alpha=0.3,
               color='steelblue', label='real')
    ax.scatter(recon_unnorm[:, 0], recon_unnorm[:, 1], s=6, alpha=0.3,
               color='darkorange', label='recon')
    ax.set_title(f'Reconstruction (MSE={recon_mse:.4f})', fontsize=10)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')
    ax.legend(markerscale=2, fontsize=9)

    # Panel 4: Code usage bar chart
    ax = axes[1, 0]
    ax.bar(range(model.K), code_counts, color=code_colors)
    ax.axhline(data.shape[0] / model.K, color='red', linestyle='--',
               label='uniform usage')
    ax.set_xlabel('Code index'); ax.set_ylabel('Usage count')
    ax.set_title(f'Codebook Usage ({active_codes}/{model.K} active)')
    ax.legend(fontsize=9)

    # Panel 5: Loss curves
    ax = axes[1, 1]
    ax.plot(history['total'], label='Total', color='steelblue')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('VQ-VAE Training Loss')
    ax.set_yscale('log')
    ax.legend()

    # Panel 6: Codebook vectors
    ax = axes[1, 2]
    # Project codebook through decoder
    decoded_codes = model.decoder.forward(model.codebook) * std
    ax.scatter(raw_data[:, 0], raw_data[:, 1], s=5, alpha=0.15, color='gray')
    ax.scatter(decoded_codes[:, 0], decoded_codes[:, 1],
               s=80, c=range(model.K), cmap='tab20', marker='*',
               edgecolors='black', linewidths=0.5, zorder=5, label='codebook centres')
    ax.set_title('Decoded Codebook Centres', fontsize=10)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal')
    ax.legend(fontsize=9)

    plt.suptitle('VQ-VAE: Pure NumPy  |  K=16 codes, D=8', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'vqvae_results.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
