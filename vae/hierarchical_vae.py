"""
vae/hierarchical_vae.py — Two-Level Hierarchical VAE (Pure NumPy)

Hierarchy
---------
  Global level:  z2 ~ p(z2) = N(0, I)
  Local level:   z1 ~ p(z1 | z2)  parameterized by a decoder MLP
  Observation:   x  ~ p(x  | z1)  parameterized by a decoder MLP

Encoder (inference)
-------------------
  q(z1 | x)    : MLP(x)  → (μ_z1, log σ²_z1)
  q(z2 | z1, x): MLP(z1) → (μ_z2, log σ²_z2)   (amortized on z1)

ELBO
----
  L = E[log p(x|z1)] - KL[q(z1|x) || p(z1|z2)] - KL[q(z2|z1) || p(z2)]

Implementation notes
--------------------
All gradients are computed analytically through stored intermediate values.
Each network is its own Mini-MLP class to avoid cache-collision issues between
multiple calls to the same layer list.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(__file__)


# ═══════════════════════════════════════ Data ═══════════════════════════════════

def make_data(n: int = 3000, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2],
                         [0, 3], [3, 0], [-3, 0], [0, -3]], dtype=float)
    k = n // len(centers)
    data = []
    for c in centers:
        data.append(rng.randn(k, 2) * 0.5 + c)
    data = np.vstack(data).astype(np.float32)
    rng.shuffle(data)
    return data


# ═══════════════════════════════════════ Mini MLP ═══════════════════════════════

class MiniMLP:
    """
    MLP: in_dim → hidden → out_dim, tanh activation.
    Carries its own forward cache.  Params: [(W1,b1),(W2,b2)].
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        s = 0.1
        self.W1 = rng.randn(in_dim, hidden).astype(np.float32) * s
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = rng.randn(hidden, out_dim).astype(np.float32) * s
        self.b2 = np.zeros(out_dim, np.float32)
        # Gradients
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
        # Adam state
        self._m = [np.zeros_like(p) for p in self._params_list()]
        self._v = [np.zeros_like(p) for p in self._params_list()]
        # Cache
        self._x: np.ndarray | None = None
        self._h: np.ndarray | None = None

    def _params_list(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _grads_list(self):
        return [self.dW1, self.db1, self.dW2, self.db2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        z1 = x @ self.W1 + self.b1
        h  = np.tanh(z1)
        self._h = h
        out = h @ self.W2 + self.b2
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (batch, out_dim) → dx: (batch, in_dim). Also sets dW1,db1,dW2,db2."""
        batch = len(dout)
        self.dW2 = self._h.T @ dout / batch
        self.db2 = dout.mean(axis=0)
        dh = dout @ self.W2.T
        dz1 = dh * (1 - self._h ** 2)   # tanh backward
        self.dW1 = self._x.T @ dz1 / batch
        self.db1 = dz1.mean(axis=0)
        return dz1 @ self.W1.T

    def adam_update(self, lr: float, t: int,
                    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        lr_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        for i, (p, g) in enumerate(zip(self._params_list(), self._grads_list())):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            p -= lr_t * self._m[i] / (np.sqrt(self._v[i]) + eps)


# ═══════════════════════════════════════ HVAE ═══════════════════════════════════

class HierarchicalVAE:
    """
    Two-level HVAE for 2D data.  Pure NumPy, analytical backward pass.
    """

    def __init__(self, x_dim: int = 2, z1_dim: int = 2, z2_dim: int = 2,
                 hidden: int = 64, lr: float = 5e-4):
        self.x_dim  = x_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.lr = lr
        self._t = 0

        # Encoder q(z1|x)
        self.enc_z1    = MiniMLP(x_dim,  hidden, hidden,  seed=1)
        self.enc_z1_mu = MiniMLP(hidden, hidden, z1_dim,  seed=2)
        self.enc_z1_lv = MiniMLP(hidden, hidden, z1_dim,  seed=3)

        # Encoder q(z2|z1)
        self.enc_z2    = MiniMLP(z1_dim, hidden, hidden,  seed=4)
        self.enc_z2_mu = MiniMLP(hidden, hidden, z2_dim,  seed=5)
        self.enc_z2_lv = MiniMLP(hidden, hidden, z2_dim,  seed=6)

        # Decoder p(z1|z2)  — outputs μ and logvar of z1 prior
        self.dec_z1    = MiniMLP(z2_dim, hidden, hidden,  seed=7)
        self.dec_z1_mu = MiniMLP(hidden, hidden, z1_dim,  seed=8)
        self.dec_z1_lv = MiniMLP(hidden, hidden, z1_dim,  seed=9)

        # Decoder p(x|z1)
        self.dec_x     = MiniMLP(z1_dim, hidden, x_dim,   seed=10)

        self._nets = [
            self.enc_z1, self.enc_z1_mu, self.enc_z1_lv,
            self.enc_z2, self.enc_z2_mu, self.enc_z2_lv,
            self.dec_z1, self.dec_z1_mu, self.dec_z1_lv,
            self.dec_x,
        ]
        self._cache: dict = {}

    # ─────────────────────────────────────── reparameterize ────────────────────
    @staticmethod
    def _reparam(mu: np.ndarray, lv: np.ndarray):
        lv_clamp = np.clip(lv, -6, 6)
        eps = np.random.randn(*mu.shape).astype(np.float32)
        std = np.exp(0.5 * lv_clamp)
        return mu + eps * std, eps, std, lv_clamp

    # ─────────────────────────────────────── forward ───────────────────────────
    def forward(self, x: np.ndarray) -> dict:
        # q(z1|x)
        h1    = self.enc_z1.forward(x)
        mu_z1 = self.enc_z1_mu.forward(h1)
        lv_z1 = self.enc_z1_lv.forward(h1)
        z1, eps1, std1, lv_z1c = self._reparam(mu_z1, lv_z1)

        # q(z2|z1)
        h2    = self.enc_z2.forward(z1)
        mu_z2 = self.enc_z2_mu.forward(h2)
        lv_z2 = self.enc_z2_lv.forward(h2)
        z2, eps2, std2, lv_z2c = self._reparam(mu_z2, lv_z2)

        # p(z1|z2)
        h3         = self.dec_z1.forward(z2)
        mu_z1_p    = self.dec_z1_mu.forward(h3)
        lv_z1_p    = np.clip(self.dec_z1_lv.forward(h3), -6, 6)

        # p(x|z1)
        mu_x = self.dec_x.forward(z1)

        self._cache = dict(
            x=x, h1=h1, mu_z1=mu_z1, lv_z1c=lv_z1c,
            z1=z1, eps1=eps1, std1=std1,
            h2=h2, mu_z2=mu_z2, lv_z2c=lv_z2c,
            z2=z2, eps2=eps2, std2=std2,
            h3=h3, mu_z1_p=mu_z1_p, lv_z1_p=lv_z1_p,
            mu_x=mu_x,
        )
        return self._cache

    # ─────────────────────────────────────── ELBO ──────────────────────────────
    def compute_elbo(self, c: dict, sigma_x: float = 0.5) -> float:
        # log p(x|z1)
        recon = -0.5 * ((c["x"] - c["mu_x"]) / sigma_x) ** 2
        log_px = recon.sum(axis=1)

        # KL[q(z1|x) || p(z1|z2)]
        var_q1 = np.exp(c["lv_z1c"])
        var_p1 = np.exp(c["lv_z1_p"])
        kl_z1 = 0.5 * (c["lv_z1_p"] - c["lv_z1c"] - 1.0
                        + var_q1 / var_p1
                        + (c["mu_z1"] - c["mu_z1_p"]) ** 2 / var_p1).sum(axis=1)

        # KL[q(z2|z1) || p(z2)=N(0,I)]
        kl_z2 = 0.5 * (-1.0 - c["lv_z2c"]
                        + c["mu_z2"] ** 2 + np.exp(c["lv_z2c"])).sum(axis=1)

        elbo = (log_px - kl_z1 - kl_z2).mean()
        return float(elbo)

    # ─────────────────────────────────────── backward ──────────────────────────
    def backward(self, sigma_x: float = 0.5):
        c = self._cache
        N = float(len(c["x"]))
        s2 = sigma_x ** 2

        # ── ∂L/∂mu_x  (reconstruction)
        d_mu_x = -(c["x"] - c["mu_x"]) / s2   # (N, x_dim)
        dz1_recon = self.dec_x.backward(d_mu_x)

        # ── KL z2 gradients
        d_mu_z2_kl = c["mu_z2"]                                # (N, z2_dim)
        d_lv_z2_kl = 0.5 * (np.exp(c["lv_z2c"]) - 1.0)

        # ── KL z1 gradients
        var_q1 = np.exp(c["lv_z1c"])
        var_p1 = np.exp(c["lv_z1_p"])
        diff1  = c["mu_z1"] - c["mu_z1_p"]

        d_mu_z1_kl  =  diff1 / var_p1
        d_lv_z1_kl  = 0.5 * (var_q1 / var_p1 - 1.0)
        # d_mu_z1_p
        d_mu_z1_p_kl = -diff1 / var_p1
        # d_lv_z1_p: ∂KL/∂lv_p = 0.5*(1 - var_q/var_p - diff²/var_p)
        d_lv_z1_p_kl = 0.5 * (1.0 - var_q1 / var_p1 - diff1 ** 2 / var_p1)

        # ── Backward through dec_z1_mu, dec_z1_lv, dec_z1
        dh3_mu  = self.dec_z1_mu.backward(d_mu_z1_p_kl)
        dh3_lv  = self.dec_z1_lv.backward(d_lv_z1_p_kl)
        dh3     = dh3_mu + dh3_lv
        dz2_kl  = self.dec_z1.backward(dh3)

        # ── Backward through enc_z2: reparam → mu/lv → h2 → z1
        # dz2 flows to mu_z2 directly and via scaled eps2*std2
        dmu_z2_reparam = dz2_kl
        dlv_z2_reparam = dz2_kl * c["eps2"] * c["std2"] * 0.5

        d_mu_z2 = d_mu_z2_kl + dmu_z2_reparam
        d_lv_z2 = d_lv_z2_kl + dlv_z2_reparam

        dh2_mu   = self.enc_z2_mu.backward(d_mu_z2)
        dh2_lv   = self.enc_z2_lv.backward(d_lv_z2)
        dh2      = dh2_mu + dh2_lv
        dz1_enc2 = self.enc_z2.backward(dh2)

        # ── Backward through enc_z1: combine all z1 gradients
        dz1_total = dz1_recon + dz1_enc2

        # reparameterization z1 = mu_z1 + eps1*std1
        dmu_z1_reparam = dz1_total
        dlv_z1_reparam = dz1_total * c["eps1"] * c["std1"] * 0.5

        d_mu_z1 = d_mu_z1_kl + dmu_z1_reparam
        d_lv_z1 = d_lv_z1_kl + dlv_z1_reparam

        dh1_mu = self.enc_z1_mu.backward(d_mu_z1)
        dh1_lv = self.enc_z1_lv.backward(d_lv_z1)
        dh1    = dh1_mu + dh1_lv
        self.enc_z1.backward(dh1)

    # ─────────────────────────────────────── train ─────────────────────────────
    def train_step(self, x: np.ndarray) -> float:
        c = self.forward(x)
        elbo = self.compute_elbo(c)
        self.backward()
        self._t += 1
        for net in self._nets:
            net.adam_update(self.lr, self._t)
        return -elbo

    def train(self, data: np.ndarray, n_epochs: int = 300,
              batch_size: int = 128) -> list[float]:
        n = len(data)
        rng = np.random.RandomState(0)
        losses: list[float] = []

        for ep in range(1, n_epochs + 1):
            idx = rng.permutation(n)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n - batch_size + 1, batch_size):
                xb = data[idx[start: start + batch_size]]
                loss = self.train_step(xb)
                ep_loss += loss
                n_batches += 1
            avg = ep_loss / max(n_batches, 1)
            losses.append(avg)
            if ep % 50 == 0 or ep == 1:
                print(f"  Epoch {ep:4d}/{n_epochs}  -ELBO = {avg:.4f}")

        return losses

    # ─────────────────────────────────────── sampling ──────────────────────────
    def sample(self, n: int = 500) -> np.ndarray:
        z2 = np.random.randn(n, self.z2_dim).astype(np.float32)
        h3      = self.dec_z1.forward(z2)
        mu_z1_p = self.dec_z1_mu.forward(h3)
        lv_z1_p = np.clip(self.dec_z1_lv.forward(h3), -6, 6)
        std1_p  = np.exp(0.5 * lv_z1_p)
        z1 = mu_z1_p + std1_p * np.random.randn(n, self.z1_dim).astype(np.float32)
        mu_x = self.dec_x.forward(z1)
        return mu_x

    def encode(self, x: np.ndarray) -> np.ndarray:
        h1    = self.enc_z1.forward(x)
        mu_z1 = self.enc_z1_mu.forward(h1)
        return mu_z1


# ═══════════════════════════════════════ Main ═══════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical VAE (2-level)")
    print("=" * 60)

    data = make_data(n=4000)
    print(f"Data shape: {data.shape},  range=[{data.min():.2f}, {data.max():.2f}]")

    # Normalize
    mu_d  = data.mean(axis=0)
    std_d = data.std(axis=0) + 1e-8
    data_norm = ((data - mu_d) / std_d).astype(np.float32)

    hvae = HierarchicalVAE(x_dim=2, z1_dim=2, z2_dim=2, hidden=64, lr=5e-4)
    losses = hvae.train(data_norm, n_epochs=300, batch_size=128)

    # ── Latent space visualization
    z1_enc = hvae.encode(data_norm[:500])

    # ── Generated samples
    x_gen = hvae.sample(500)

    # ── Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.scatter(data_norm[:500, 0], data_norm[:500, 1],
               s=8, alpha=0.5, c="steelblue", label="Real")
    ax.scatter(x_gen[:, 0], x_gen[:, 1],
               s=8, alpha=0.5, c="darkorange", label="Generated")
    ax.set_title("Real vs Generated")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    colors_label = np.tile(np.arange(8), 500 // 8 + 1)[:500]
    ax.scatter(z1_enc[:, 0], z1_enc[:, 1],
               s=8, alpha=0.6, c=colors_label, cmap="tab10")
    ax.set_title("Latent space z1 (μ)")
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(losses, color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("-ELBO")
    ax.set_title("Training loss")
    ax.grid(alpha=0.2)

    plt.suptitle("Hierarchical VAE", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "hvae_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nResults saved → {out}")
    print(f"Final -ELBO: {losses[-1]:.4f}")
