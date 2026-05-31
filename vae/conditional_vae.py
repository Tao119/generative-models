"""
Conditional VAE (CVAE) — pure NumPy.

Architecture
------------
  One-hot label (4-dim) concatenated to input (2-dim) → 6-dim input to encoder
  Encoder: Affine(6→64)→ReLU→Affine(64→32)→ReLU → mu(32→2), logvar(32→2)
  Decoder: Affine(6→32)→ReLU→Affine(32→64)→ReLU→Affine(64→2)
            where input = concat(z[2], label[4]) = 6-dim

Data: 2D Gaussian mixture — 4 clusters at corners of a square
      centers: (2,2), (-2,2), (-2,-2), (2,-2)

Loss: MSE reconstruction + 0.5 * KL divergence (beta=1)
Optimizer: Adam lr=1e-3, 500 epochs, batch=64
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

NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_gaussian_mixture(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]], dtype=float)
    n_each = n // NUM_CLASSES
    data, labels = [], []
    for i, c in enumerate(centers):
        pts = rng.randn(n_each, 2) * 0.4 + c
        data.append(pts)
        labels.extend([i] * n_each)
    return np.vstack(data), np.array(labels)


def to_onehot(labels, num_classes=NUM_CLASSES):
    n = len(labels)
    oh = np.zeros((n, num_classes), dtype=float)
    oh[np.arange(n), labels] = 1.0
    return oh


# ---------------------------------------------------------------------------
# Conditional Encoder
# ---------------------------------------------------------------------------

class ConditionalEncoder:
    """Input: concat(x[2], label[4]) = 6-dim"""

    def __init__(self, latent_dim=2):
        in_dim = 2 + NUM_CLASSES  # 6
        W1 = he_init(in_dim, 64)
        W2 = he_init(64, 32)
        W_mu = he_init(32, latent_dim)
        W_lv = he_init(32, latent_dim)

        self.fc1 = Affine(W1, np.zeros(64))
        self.relu1 = Relu()
        self.fc2 = Affine(W2, np.zeros(32))
        self.relu2 = Relu()
        self.fc_mu = Affine(W_mu, np.zeros(latent_dim))
        self.fc_lv = Affine(W_lv, np.zeros(latent_dim))

        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2]

    def forward(self, x, label_oh):
        h = np.concatenate([x, label_oh], axis=1)
        for layer in self.layers:
            h = layer.forward(h)
        mu = self.fc_mu.forward(h)
        logvar = self.fc_lv.forward(h)
        return mu, logvar

    def backward(self, dmu, dlogvar):
        dh = self.fc_mu.backward(dmu) + self.fc_lv.backward(dlogvar)
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        # dh has shape (N, 6); first 2 dims = dx, last 4 = d_label (ignored)
        return dh

    def params_and_grads(self):
        params, grads = [], []
        for layer in self.layers + [self.fc_mu, self.fc_lv]:
            params += layer.params
            grads += layer.grads
        return params, grads


# ---------------------------------------------------------------------------
# Conditional Decoder
# ---------------------------------------------------------------------------

class ConditionalDecoder:
    """Input: concat(z[2], label[4]) = 6-dim"""

    def __init__(self, latent_dim=2):
        in_dim = latent_dim + NUM_CLASSES  # 6
        W1 = he_init(in_dim, 32)
        W2 = he_init(32, 64)
        W3 = he_init(64, 2)

        self.fc1 = Affine(W1, np.zeros(32))
        self.relu1 = Relu()
        self.fc2 = Affine(W2, np.zeros(64))
        self.relu2 = Relu()
        self.fc3 = Affine(W3, np.zeros(2))

        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3]

    def forward(self, z, label_oh):
        h = np.concatenate([z, label_oh], axis=1)
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, dout):
        dh = dout
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        # dh: (N, latent_dim + NUM_CLASSES); return only the z-part
        return dh[:, :2]   # gradient w.r.t. z (label gradient discarded)

    def params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads
        return params, grads


# ---------------------------------------------------------------------------
# CVAE
# ---------------------------------------------------------------------------

class CVAE:
    def __init__(self, latent_dim=2, beta=1.0, lr=1e-3):
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = ConditionalEncoder(latent_dim)
        self.decoder = ConditionalDecoder(latent_dim)
        self.optimizer = Adam(lr=lr)
        self._eps = None

    def forward(self, x, label_oh):
        mu, logvar = self.encoder.forward(x, label_oh)
        self._eps = np.random.randn(*mu.shape)
        z = mu + self._eps * np.exp(0.5 * logvar)
        x_recon = self.decoder.forward(z, label_oh)
        return x_recon, mu, logvar, z

    def loss(self, x, x_recon, mu, logvar):
        N = x.shape[0]
        recon = np.sum((x - x_recon) ** 2) / N
        kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar)) / N
        return recon + self.beta * kl, recon, kl

    def backward(self, x, x_recon, mu, logvar, label_oh):
        N = x.shape[0]
        # Reconstruction gradient
        dx_recon = -2.0 * (x - x_recon) / N
        # dz from decoder
        dz = self.decoder.backward(dx_recon)

        # KL gradients
        dmu_kl = self.beta * mu / N
        dlogvar_kl = self.beta * 0.5 * (np.exp(logvar) - 1) / N

        # Reparameterisation: z = mu + eps * exp(0.5*logvar)
        # dL/dmu = dL/dz * dz/dmu = dz * 1
        # dL/dlogvar = dL/dz * dz/dlogvar = dz * eps * 0.5 * exp(0.5*logvar)
        dmu = dmu_kl + dz
        dlogvar = dlogvar_kl + dz * self._eps * 0.5 * np.exp(0.5 * logvar)

        self.encoder.backward(dmu, dlogvar)

    def step(self, x, label_oh):
        x_recon, mu, logvar, z = self.forward(x, label_oh)
        total, recon, kl = self.loss(x, x_recon, mu, logvar)
        self.backward(x, x_recon, mu, logvar, label_oh)

        enc_p, enc_g = self.encoder.params_and_grads()
        dec_p, dec_g = self.decoder.params_and_grads()
        self.optimizer.update(enc_p + dec_p, enc_g + dec_g)
        return total, recon, kl

    def generate(self, label_id, n=50):
        """Sample n points conditioned on a class label."""
        z = np.random.randn(n, self.latent_dim)
        label_oh = np.zeros((n, NUM_CLASSES))
        label_oh[:, label_id] = 1.0
        return self.decoder.forward(z, label_oh)

    def encode(self, x, label_oh):
        return self.encoder.forward(x, label_oh)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, data, labels, epochs=500, batch_size=64):
    N = data.shape[0]
    label_oh = to_onehot(labels)
    losses = []
    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            sel = idx[i:i + batch_size]
            xb = data[sel]
            lb = label_oh[sel]
            loss, _, _ = model.step(xb, lb)
            total_loss += loss
            n_batches += 1
        losses.append(total_loss / n_batches)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  ELBO: {losses[-1]:.4f}")
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(0)
    t0 = time.time()

    data, labels = make_gaussian_mixture(n=1000)
    label_oh = to_onehot(labels)

    model = CVAE(latent_dim=2, beta=1.0, lr=1e-3)
    losses = train(model, data, labels, epochs=500, batch_size=64)

    elapsed = time.time() - t0
    print(f"\nFinal ELBO: {losses[-1]:.4f}  (runtime: {elapsed:.1f}s)")

    # Per-class reconstruction MSE
    print("\n--- Per-class reconstruction MSE ---")
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        xc = data[mask]
        lc = label_oh[mask]
        mu, logvar = model.encode(xc, lc)
        z = mu  # use mean (no sampling noise for evaluation)
        x_recon = model.decoder.forward(z, lc)
        mse = np.mean((xc - x_recon) ** 2)
        print(f"  Class {cls}: MSE = {mse:.4f}")

    # Generation demo
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]])

    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(16, 4))
    for cls in range(NUM_CLASSES):
        ax = axes[cls]
        # Real points
        mask = labels == cls
        ax.scatter(data[mask, 0], data[mask, 1], s=15, alpha=0.4,
                   color=colors[cls], label='real')
        # Generated points
        gen = model.generate(cls, n=50)
        ax.scatter(gen[:, 0], gen[:, 1], s=25, alpha=0.8,
                   color='black', marker='x', label='generated')
        ax.set_title(f'Class {cls}  (center≈{tuple(centers[cls])})')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.legend(markerscale=1.5, fontsize=8)

    plt.suptitle('Conditional VAE — Real vs Generated per Class', fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cvae_results.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
