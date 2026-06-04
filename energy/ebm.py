"""
energy/ebm.py — Energy-Based Model with Contrastive Divergence

Architecture
------------
E(x; θ): scalar-valued MLP
  x (2D) → Linear(2, 128) → LeakyReLU → Linear(128, 128) → LeakyReLU → Linear(128, 1)

Training
--------
Contrastive Divergence (CD-k, k=10):
  1. Sample negative particles via k steps of Langevin dynamics from noise.
  2. Loss = E[E(x+)] - E[E(x-)]   (positive ← data,  negative ← MCMC)
  3. Gradient descent on θ.

Sampling (at eval)
------------------
Langevin dynamics:
  x_{t+1} = x_t - ε ∇_x E(x_t) + √(2ε) η,  η ~ N(0, I)

Dataset
-------
2D mixture of 8 Gaussians.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.layers import Affine, Adam, he_init

OUT_DIR = os.path.dirname(__file__)


# ═══════════════════════════════════════ Data ═══════════════════════════════════

def make_8_gaussians(n: int = 3000, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    radius = 2.5
    centers = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    k = n // 8
    data = []
    for c in centers:
        data.append(rng.randn(k, 2) * 0.3 + c)
    data = np.vstack(data).astype(np.float32)
    rng.shuffle(data)
    return data


# ═══════════════════════════════════════ Leaky ReLU layer ═══════════════════════

class LeakyRelu:
    def __init__(self, alpha: float = 0.2):
        self.params, self.grads = [], []
        self.alpha = alpha
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return np.where(self._x > 0, dout, self.alpha * dout)


# ═══════════════════════════════════════ Energy Network ═════════════════════════

class EnergyNet:
    """
    MLP: 2 → 128 → 128 → 1  (scalar energy).
    Supports forward + backward via manual autograd.
    """

    def __init__(self, hidden: int = 128, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = 0.01
        W1 = rng.randn(2, hidden).astype(np.float32) * scale
        W2 = rng.randn(hidden, hidden).astype(np.float32) * scale
        W3 = rng.randn(hidden, 1).astype(np.float32) * scale
        b1 = np.zeros(hidden, np.float32)
        b2 = np.zeros(hidden, np.float32)
        b3 = np.zeros(1, np.float32)

        self.layers = [
            Affine(W1, b1), LeakyRelu(),
            Affine(W2, b2), LeakyRelu(),
            Affine(W3, b3),
        ]

        self.params = []
        self.grads  = []
        for l in self.layers:
            self.params.extend(l.params)
            self.grads.extend(l.grads)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, 2) → (batch, 1)"""
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (batch, 1)"""
        d = dout
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def grad_wrt_input(self, x: np.ndarray) -> np.ndarray:
        """
        ∇_x E(x): gradient of energy w.r.t. input, via finite differences.
        (Avoids storing per-sample Jacobians.)
        """
        eps = 1e-4
        grad = np.zeros_like(x)
        for j in range(x.shape[1]):
            xp = x.copy(); xp[:, j] += eps
            xm = x.copy(); xm[:, j] -= eps
            ep = self.forward(xp).squeeze(1)
            em = self.forward(xm).squeeze(1)
            grad[:, j] = (ep - em) / (2 * eps)
        return grad


# ═══════════════════════════════════════ Langevin MCMC ══════════════════════════

def langevin_dynamics(net: EnergyNet, x_init: np.ndarray,
                      n_steps: int = 10, step_size: float = 0.1,
                      noise_scale: float = 0.005,
                      clip_range: float = 6.0) -> np.ndarray:
    """
    x_{t+1} = x_t - ε ∇_x E(x_t) + √(2ε) η
    """
    x = x_init.copy().astype(np.float32)
    for _ in range(n_steps):
        grad = net.grad_wrt_input(x)
        noise = np.random.randn(*x.shape).astype(np.float32)
        x = x - step_size * grad + np.sqrt(2 * step_size) * noise_scale * noise
        x = np.clip(x, -clip_range, clip_range)
    return x


# ═══════════════════════════════════════ EBM trainer ════════════════════════════

class EBMTrainer:
    def __init__(self, hidden: int = 128, lr: float = 1e-4,
                 cd_k: int = 10, step_size: float = 0.1,
                 noise_scale: float = 0.005):
        self.net = EnergyNet(hidden=hidden)
        self.cd_k = cd_k
        self.step_size = step_size
        self.noise_scale = noise_scale
        self._opt = Adam(lr=lr)

    def _cd_step(self, x_pos: np.ndarray) -> tuple[float, np.ndarray]:
        """
        One CD-k step.
        Returns (loss, x_neg).
        """
        batch = len(x_pos)
        # ── Negative phase: start from noise, run k Langevin steps
        x_neg_init = np.random.randn(*x_pos.shape).astype(np.float32) * 2.0
        x_neg = langevin_dynamics(self.net, x_neg_init,
                                   n_steps=self.cd_k,
                                   step_size=self.step_size,
                                   noise_scale=self.noise_scale)

        # ── Positive phase: forward + backward to get ∂E_pos/∂θ
        self.net.forward(x_pos)
        self.net.backward(np.ones((batch, 1), np.float32) / batch)
        grads_pos = [g.copy() for g in self.net.grads]
        e_pos_val = self.net.layers[-1].x.mean()   # approximate from affine cache

        # ── Negative phase: forward + backward to get ∂E_neg/∂θ
        e_neg_out = self.net.forward(x_neg)
        self.net.backward(np.ones((batch, 1), np.float32) / batch)
        grads_neg = [g.copy() for g in self.net.grads]

        # ── CD loss = E[E(x+)] - E[E(x-)]
        loss = float(grads_pos[0].mean() - grads_neg[0].mean())   # proxy
        # Directly compute from energies
        e_pos = self.net.forward(x_pos)
        e_neg_arr = self.net.forward(x_neg)
        loss = float(e_pos.mean() - e_neg_arr.mean())

        # ── Combined gradient = ∂E_pos/∂θ - ∂E_neg/∂θ
        for g, gp, gn in zip(self.net.grads, grads_pos, grads_neg):
            g[...] = gp - gn

        self._opt.update(self.net.params, self.net.grads)
        return loss, x_neg

    def train(self, data: np.ndarray, n_epochs: int = 100,
              batch_size: int = 128) -> list[float]:
        n = len(data)
        rng = np.random.RandomState(42)
        losses: list[float] = []

        for ep in range(1, n_epochs + 1):
            idx = rng.permutation(n)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n - batch_size + 1, batch_size):
                xb = data[idx[start: start + batch_size]]
                loss, _ = self._cd_step(xb)
                ep_loss += loss
                n_batches += 1
            avg = ep_loss / max(n_batches, 1)
            losses.append(avg)
            if ep % 20 == 0 or ep == 1:
                print(f"  Epoch {ep:4d}/{n_epochs}  CD loss = {avg:.4f}")

        return losses

    def sample(self, n: int = 500, n_steps: int = 500) -> np.ndarray:
        x_init = np.random.randn(n, 2).astype(np.float32) * 2.0
        return langevin_dynamics(self.net, x_init, n_steps=n_steps,
                                  step_size=self.step_size,
                                  noise_scale=self.noise_scale)


# ═══════════════════════════════════════ Visualization ══════════════════════════

def plot_energy_landscape(trainer: EBMTrainer, data: np.ndarray,
                           samples: np.ndarray, losses: list[float],
                           out_dir: str) -> None:
    # Build grid for energy contour
    grid_res = 50
    x1 = np.linspace(-4, 4, grid_res)
    x2 = np.linspace(-4, 4, grid_res)
    XX, YY = np.meshgrid(x1, x2)
    grid = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)

    with_energy = trainer.net.forward(grid).reshape(grid_res, grid_res)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Energy landscape
    ax = axes[0]
    cf = ax.contourf(XX, YY, with_energy, levels=30, cmap="RdYlBu_r")
    ax.scatter(data[:300, 0], data[:300, 1],
               s=5, c="white", alpha=0.5, label="Data")
    plt.colorbar(cf, ax=ax)
    ax.set_title("Learned Energy Landscape")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(fontsize=7)

    # ── Samples vs data
    ax = axes[1]
    ax.scatter(data[:300, 0], data[:300, 1],
               s=8, alpha=0.5, c="steelblue", label="Data")
    ax.scatter(samples[:, 0], samples[:, 1],
               s=8, alpha=0.5, c="darkorange", label="Langevin samples")
    ax.set_title("Data vs Generated Samples")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ── Training loss
    ax = axes[2]
    ax.plot(losses, color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CD Loss")
    ax.set_title("Contrastive Divergence Training Loss")
    ax.grid(alpha=0.2)

    plt.suptitle("Energy-Based Model (CD-k + Langevin)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "ebm_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Results saved → {out}")


# ═══════════════════════════════════════ Main ═══════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Energy-Based Model — CD-10 + Langevin Dynamics")
    print("=" * 60)

    data = make_8_gaussians(n=3000)
    print(f"Data: {data.shape}  range=[{data.min():.2f}, {data.max():.2f}]")

    trainer = EBMTrainer(
        hidden=128,
        lr=1e-4,
        cd_k=10,
        step_size=0.05,
        noise_scale=0.01,
    )
    print("\nTraining EBM …")
    losses = trainer.train(data, n_epochs=100, batch_size=128)

    print("\nSampling via Langevin dynamics …")
    samples = trainer.sample(n=500, n_steps=300)
    print(f"Generated samples: {samples.shape}")

    plot_energy_landscape(trainer, data, samples, losses, OUT_DIR)
    print(f"Final CD loss: {losses[-1]:.4f}")
    print("Done.")
