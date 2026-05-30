import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.layers import Affine, Relu, Adam, he_init


def make_gaussian_mixture(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]], dtype=float)
    n_each = n // 4
    data = []
    for c in centers:
        data.append(rng.randn(n_each, 2) * 0.4 + c)
    return np.vstack(data)


def sinusoidal_embedding(t, dim=16):
    half = dim // 2
    freqs = np.exp(-np.log(10000) * np.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)


class NoiseNet:
    def __init__(self, t_dim=16):
        in_dim = 2 + t_dim
        self.fc1 = Affine(he_init(in_dim, 64), np.zeros(64))
        self.relu1 = Relu()
        self.fc2 = Affine(he_init(64, 64), np.zeros(64))
        self.relu2 = Relu()
        self.fc3 = Affine(he_init(64, 2), np.zeros(2))
        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3]
        self.t_dim = t_dim
        self._inp = None

    def forward(self, xt, t_emb):
        self._inp = np.concatenate([xt, t_emb], axis=1)
        h = self._inp
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, dout):
        dh = dout
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        return dh[:, :2]

    def params_and_grads(self):
        params, grads = [], []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads
        return params, grads


class DDPM:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, t_dim=16, lr=1e-3):
        self.T = T
        self.t_dim = t_dim
        self.beta = np.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        self.net = NoiseNet(t_dim=t_dim)
        self.optimizer = Adam(lr=lr)

    def q_sample(self, x0, t):
        ab = self.alpha_bar[t][:, None]
        eps = np.random.randn(*x0.shape)
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        return xt, eps

    def loss_and_backward(self, x0, t):
        xt, eps = self.q_sample(x0, t)
        t_emb = sinusoidal_embedding(t.astype(float), dim=self.t_dim)
        eps_pred = self.net.forward(xt, t_emb)
        N = x0.shape[0]
        diff = eps_pred - eps
        loss = np.sum(diff ** 2) / N
        dout = 2 * diff / N
        self.net.backward(dout)
        return loss

    def step(self, x0, t):
        loss = self.loss_and_backward(x0, t)
        params, grads = self.net.params_and_grads()
        self.optimizer.update(params, grads)
        return loss

    @property
    def _snap_steps(self):
        return [self.T - 1, self.T // 2, self.T // 4, 0]

    def sample(self, n, collect_steps=None):
        if collect_steps is None:
            collect_steps = set()
        x = np.random.randn(n, 2)
        collected = {}
        for t_idx in reversed(range(self.T)):
            t_arr = np.full(n, t_idx, dtype=int)
            t_emb = sinusoidal_embedding(t_arr.astype(float), dim=self.t_dim)
            eps_pred = self.net.forward(x, t_emb)

            ab = self.alpha_bar[t_idx]
            a = self.alpha[t_idx]
            b = self.beta[t_idx]

            x0_pred = (x - np.sqrt(1 - ab) * eps_pred) / np.sqrt(ab)
            x0_pred = np.clip(x0_pred, -4, 4)

            mean = (np.sqrt(a) * (1 - ab / a) * x + np.sqrt(ab / a) * (1 - a) * x0_pred) / (1 - ab)
            if t_idx > 0:
                noise = np.random.randn(*x.shape)
                x = mean + np.sqrt(b) * noise
            else:
                x = mean

            if t_idx in collect_steps:
                collected[t_idx] = x.copy()

        return x, collected


def train(model, data, epochs=1000, batch_size=128):
    N = data.shape[0]
    losses = []
    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        total = 0.0
        n_b = 0
        for i in range(0, N - batch_size + 1, batch_size):
            xb = data[idx[i:i + batch_size]]
            t = np.random.randint(0, model.T, size=xb.shape[0])
            loss = model.step(xb, t)
            total += loss
            n_b += 1
        losses.append(total / max(n_b, 1))
    return losses


def main():
    np.random.seed(0)
    t0 = time.time()

    raw = make_gaussian_mixture(n=1000)
    std = raw.std(axis=0)
    data = raw / std

    model = DDPM(T=200, beta_start=1e-4, beta_end=0.02, t_dim=16, lr=1e-3)
    losses = train(model, data, epochs=1000, batch_size=128)

    elapsed = time.time() - t0
    print(f"Final training loss: {losses[-1]:.6f}  (runtime: {elapsed:.1f}s)")

    collect_steps = {199, 100, 50, 0}
    samples, collected = model.sample(500, collect_steps=collect_steps)

    step_keys = sorted(collected.keys(), reverse=True)
    n_panels = len(step_keys) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    real_unnorm = data * std

    for ax, sk in zip(axes[:-1], step_keys):
        pts = collected[sk] * std
        label = f't={sk}' if sk > 0 else 'final'
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, color='tab:purple')
        ax.set_title(f'Denoising step {label}')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')

    ax_final = axes[-1]
    final_unnorm = samples * std
    ax_final.scatter(raw[:, 0], raw[:, 1], s=8, alpha=0.3, label='real')
    ax_final.scatter(final_unnorm[:, 0], final_unnorm[:, 1], s=8, alpha=0.5,
                     color='tab:red', label='generated')
    ax_final.set_title('Final vs Real')
    ax_final.set_xlim(-5, 5)
    ax_final.set_ylim(-5, 5)
    ax_final.set_aspect('equal')
    ax_final.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'ddpm_results.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(losses)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('DDPM Training Loss')
    plt.tight_layout()
    curve_path = os.path.join(os.path.dirname(__file__), 'ddpm_loss_curve.png')
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"Saved {curve_path}")


if __name__ == '__main__':
    main()
