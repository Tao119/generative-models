"""
flow/normalizing_flow.py — RealNVP Normalizing Flow (Pure NumPy)

Architecture
------------
4 coupling layers with alternating binary masks.
Each coupling layer:
  z  = [z1, z2]   (split by mask)
  z2' = z2 ⊙ exp(s(z1)) + t(z1)
  log|det J| = sum(s(z1))

s(·) and t(·) are simple MLPs (NumPy, tanh activations).

Training
--------
Minimise negative log-likelihood = -E[log p_z(z) + Σ log|det J_i|]
Stochastic gradient descent with numerical gradients (finite differences)
for simplicity.

Actually we implement full analytical forward/backward pass.

Datasets
--------
- 2D Banana / crescent
- 8-Gaussian mixture
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(__file__)


# ═══════════════════════════════════════ Data generators ════════════════════════

def make_8_gaussians(n: int = 2000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    radius = 3.0
    centers = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    idx = rng.integers(0, 8, size=n)
    data = centers[idx] + rng.standard_normal((n, 2)) * 0.4
    return data.astype(np.float32)


def make_banana(n: int = 2000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n) * 1.5
    x2 = x1 ** 2 - 3 + rng.standard_normal(n) * 0.5
    return np.column_stack([x1, x2]).astype(np.float32)


# ═══════════════════════════════════════ MLP helpers ════════════════════════════

class MLP:
    """
    Tiny MLP: input_dim → hidden → hidden → output_dim
    Activation: tanh.  Trainable params stored as list of (W, b).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 0.01
        self.params: list[tuple[np.ndarray, np.ndarray]] = [
            (rng.standard_normal((input_dim,  hidden_dim)).astype(np.float32) * scale,
             np.zeros(hidden_dim, dtype=np.float32)),
            (rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * scale,
             np.zeros(hidden_dim, dtype=np.float32)),
            (rng.standard_normal((hidden_dim, output_dim)).astype(np.float32) * scale,
             np.zeros(output_dim, dtype=np.float32)),
        ]
        self._cache: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, input_dim) → (batch, output_dim).  Caches for backward."""
        self._cache = [x]
        h = x
        for i, (W, b) in enumerate(self.params):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.params) - 1 else z
            self._cache.append(z if i < len(self.params) - 1 else h)
        return h

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, list]:
        """Returns (dx, grads_list) where grads_list matches self.params."""
        grads = []
        d = dout
        for i in reversed(range(len(self.params))):
            W, b = self.params[i]
            h_in = self._cache[i]
            if i < len(self.params) - 1:
                z = self._cache[i + 1]
                d = d * (1 - np.tanh(z) ** 2)   # tanh backward
            dW = h_in.T @ d
            db = d.sum(axis=0)
            dx = d @ W.T
            grads.insert(0, (dW, db))
            d = dx
        return d, grads

    def all_params(self) -> list[np.ndarray]:
        return [p for wb in self.params for p in wb]

    def all_grads_flat(self, grads_list: list) -> list[np.ndarray]:
        return [g for gw, gb in grads_list for g in (gw, gb)]


# ═══════════════════════════════════════ Coupling layer ═════════════════════════

class CouplingLayer:
    """
    RealNVP coupling layer.
    mask: binary array of shape (2,), 1 = pass-through.
    """
    def __init__(self, mask: np.ndarray, hidden: int = 64, seed: int = 0):
        self.mask = mask.astype(np.float32)
        d_in = int(mask.sum())
        self.s_net = MLP(d_in, hidden, d_in, seed=seed)
        self.t_net = MLP(d_in, hidden, d_in, seed=seed + 1)
        self._cache: dict = {}

    def forward(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        z: (batch, 2)
        Returns (z_out, log_det_J) where log_det_J: (batch,)
        """
        z1 = z[:, self.mask == 1]   # pass-through part
        z2 = z[:, self.mask == 0]   # transformed part

        s = self.s_net.forward(z1)
        t = self.t_net.forward(z1)
        # Clamp s to avoid explosion
        s = np.tanh(s) * 2.0

        z2_out = z2 * np.exp(s) + t
        log_det = s.sum(axis=1)

        z_out = np.zeros_like(z)
        z_out[:, self.mask == 1] = z1
        z_out[:, self.mask == 0] = z2_out

        self._cache = {"z1": z1, "z2": z2, "s": s, "t": t, "z2_out": z2_out}
        return z_out, log_det

    def inverse(self, z_out: np.ndarray) -> np.ndarray:
        """Invert the coupling layer (for sampling)."""
        z1 = z_out[:, self.mask == 1]
        z2_out = z_out[:, self.mask == 0]

        s = np.tanh(self.s_net.forward(z1)) * 2.0
        t = self.t_net.forward(z1)
        z2 = (z2_out - t) * np.exp(-s)

        z = np.zeros_like(z_out)
        z[:, self.mask == 1] = z1
        z[:, self.mask == 0] = z2
        return z

    def backward(self, dz_out: np.ndarray,
                 d_log_det: np.ndarray) -> tuple[np.ndarray, list, list]:
        """
        Returns (dz_in, s_grads, t_grads).
        d_log_det: (batch,) gradient from upstream w.r.t. log_det.
        """
        z1   = self._cache["z1"]
        z2   = self._cache["z2"]
        s    = self._cache["s"]
        t    = self._cache["t"]
        z2_out = self._cache["z2_out"]

        # Gradient w.r.t. z2_out
        dz2_out = dz_out[:, self.mask == 0]
        # z2_out = z2 * exp(s) + t  →  dz2 = dz2_out * exp(s)
        dz2 = dz2_out * np.exp(s)

        # Gradient w.r.t. s: from transform + from log_det
        # ∂loss/∂s_j = dz2_out_j * z2_j * exp(s_j) + d_log_det
        ds = dz2_out * z2 * np.exp(s) + d_log_det[:, np.newaxis]
        # Through tanh clamp: s = tanh(s_raw)*2  → ds_raw = ds * 2 * (1-tanh²)
        s_raw_tanh = s / 2.0
        ds_raw = ds * 2.0 * (1 - s_raw_tanh ** 2)

        # Gradient w.r.t. t: dz2_out passes straight
        dt = dz2_out

        # Backward through s_net and t_net
        _, s_grads_list = self.s_net.backward(ds_raw)
        s_dx = self.s_net._cache  # already consumed in backward
        _, t_grads_list = self.t_net.backward(dt)

        # z1 contribution: grads from both nets
        dz1_from_s = self.s_net.backward(ds_raw)[0]
        dz1_from_t = self.t_net.backward(dt)[0]
        dz1 = dz1_from_s + dz1_from_t + dz_out[:, self.mask == 1]

        dz_in = np.zeros_like(dz_out)
        dz_in[:, self.mask == 1] = dz1
        dz_in[:, self.mask == 0] = dz2

        return dz_in, s_grads_list, t_grads_list


# ═══════════════════════════════════════ RealNVP model ══════════════════════════

class RealNVP:
    """4 coupling layers with alternating masks, 2D → 2D."""

    def __init__(self, n_layers: int = 4, hidden: int = 64, seed: int = 42):
        masks = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, 1]),
        ][:n_layers]
        self.layers = [
            CouplingLayer(m, hidden=hidden, seed=seed + i)
            for i, m in enumerate(masks)
        ]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x: (batch, 2) data → z (batch, 2) latent.
        Returns (z, total_log_det_J).
        """
        z = x.copy()
        total_log_det = np.zeros(len(x))
        for layer in self.layers:
            z, ldj = layer.forward(z)
            total_log_det += ldj
        return z, total_log_det

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """z → x by inverting all layers in reverse."""
        x = z.copy()
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Per-sample log-likelihood under standard normal prior."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2).sum(axis=1) - np.log(2 * np.pi)
        return log_pz + log_det

    def nll(self, x: np.ndarray) -> float:
        """Mean negative log-likelihood."""
        return float(-self.log_likelihood(x).mean())

    def all_params(self) -> list[np.ndarray]:
        params = []
        for layer in self.layers:
            params.extend(layer.s_net.all_params())
            params.extend(layer.t_net.all_params())
        return params


# ═══════════════════════════════════════ Training (Adam) ════════════════════════

class AdamOptimizer:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_realnvp(model: RealNVP, data: np.ndarray,
                  n_epochs: int = 200, batch_size: int = 256,
                  lr: float = 1e-3, seed: int = 42) -> list[float]:
    """
    Trains using finite-difference numerical gradients (simple, no autograd).
    Sufficient for 2D experiments.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    loss_curve: list[float] = []
    eps_fd = 1e-4   # finite-difference step

    params = model.all_params()
    opt = AdamOptimizer(params, lr=lr)

    for epoch in range(1, n_epochs + 1):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            xb = data[idx[start: start + batch_size]].astype(np.float32)
            base_loss = model.nll(xb)

            grads = []
            for p in params:
                g = np.zeros_like(p)
                flat = p.ravel()
                g_flat = g.ravel()
                for j in range(len(flat)):
                    orig = flat[j]
                    flat[j] = orig + eps_fd
                    loss_p = model.nll(xb)
                    flat[j] = orig - eps_fd
                    loss_m = model.nll(xb)
                    flat[j] = orig
                    g_flat[j] = (loss_p - loss_m) / (2 * eps_fd)
                grads.append(g)

            opt.step(grads)
            epoch_loss += base_loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_curve.append(avg_loss)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  NLL = {avg_loss:.4f}")

    return loss_curve


# ═══════════════════════════════════════ Visualization ══════════════════════════

def plot_results(model: RealNVP, data: np.ndarray, dataset_name: str,
                 loss_curve: list[float], out_dir: str) -> None:
    # Generate samples by inverting from N(0,I)
    rng = np.random.default_rng(0)
    z_samples = rng.standard_normal((1000, 2)).astype(np.float32)
    x_samples = model.inverse(z_samples)

    # ── side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.4, color="steelblue")
    ax.set_title("Training data")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.scatter(x_samples[:, 0], x_samples[:, 1], s=5, alpha=0.4, color="darkorange")
    ax.set_title("Generated samples (RealNVP)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(loss_curve, color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.set_title("Training loss")
    ax.grid(alpha=0.2)

    plt.suptitle(f"RealNVP — {dataset_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"normalizing_flow_{dataset_name}_results.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Results plot saved → {path}")

    # ── loss curve separately
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(loss_curve, color="crimson")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Negative Log-Likelihood")
    ax2.set_title(f"RealNVP Training Loss — {dataset_name}")
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, f"normalizing_flow_{dataset_name}_loss.png")
    fig2.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Loss curve saved → {loss_path}")


# ═══════════════════════════════════════ Main ═══════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RealNVP Normalizing Flow")
    print("=" * 60)

    for ds_name, ds_fn in [("8gaussians", make_8_gaussians),
                            ("banana", make_banana)]:
        print(f"\nDataset: {ds_name}")
        data = ds_fn(n=3000)

        # Normalize data
        mu = data.mean(axis=0)
        sigma = data.std(axis=0) + 1e-8
        data_norm = (data - mu) / sigma

        model = RealNVP(n_layers=4, hidden=32, seed=42)
        print(f"  Training RealNVP ({len(model.layers)} coupling layers) …")
        loss_curve = train_realnvp(
            model, data_norm, n_epochs=100, batch_size=256, lr=5e-4
        )
        print(f"  Final NLL: {loss_curve[-1]:.4f}")
        plot_results(model, data_norm, ds_name, loss_curve, OUT_DIR)

    print("\nDone.")
