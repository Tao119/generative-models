import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.layers import Affine, Relu, LeakyRelu, Sigmoid, Tanh, Adam, he_init


def make_gaussian_mixture(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]], dtype=float)
    n_each = n // 4
    data = []
    for c in centers:
        data.append(rng.randn(n_each, 2) * 0.4 + c)
    return np.vstack(data)


class Generator:
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


class Discriminator:
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


def train(real_data, z_dim=8, epochs=2000, batch_size=64, lr=2e-4):
    N = real_data.shape[0]
    G = Generator(z_dim)
    D = Discriminator()
    opt_G = Adam(lr=lr, beta1=0.5)
    opt_D = Adam(lr=lr, beta1=0.5)

    snap_epochs = {200, 500, 1000, 2000}
    snapshots = {}

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        n_batches = 0

        for i in range(0, N - batch_size + 1, batch_size):
            xb = real_data[idx[i:i + batch_size]]
            z = np.random.randn(batch_size, z_dim)
            fake = G.forward(z)

            # D step: accumulate grads from real and fake in one pass per batch
            D.zero_grads()

            d_real = D.forward(xb)
            loss_real, grad_real = bce_grad(d_real, np.ones((batch_size, 1)))
            D.backward(grad_real)
            d_grads_real = [g.copy() for _, g in zip(*D.params_and_grads())]

            # We can't easily split grads without re-zeroing, so use concatenated batch
            combined = np.concatenate([xb, fake], axis=0)
            combined_labels = np.concatenate([
                np.ones((batch_size, 1)),
                np.zeros((batch_size, 1))
            ], axis=0)
            D.zero_grads()
            d_combined = D.forward(combined)
            d_loss, d_grad = bce_grad(d_combined, combined_labels)
            D.backward(d_grad)
            d_params, d_grads = D.params_and_grads()
            opt_D.update(d_params, d_grads)

            # G step
            z2 = np.random.randn(batch_size, z_dim)
            fake2 = G.forward(z2)
            D.zero_grads()
            d_fake2 = D.forward(fake2)
            g_loss, g_grad = bce_grad(d_fake2, np.ones((batch_size, 1)))
            d_inp_grad = D.backward(g_grad)
            G.backward(d_inp_grad)
            g_params, g_grads = G.params_and_grads()
            opt_G.update(g_params, g_grads)

            d_loss_sum += d_loss
            g_loss_sum += g_loss
            n_batches += 1

        if epoch % 500 == 0:
            d_avg = d_loss_sum / max(n_batches, 1)
            g_avg = g_loss_sum / max(n_batches, 1)
            print(f"Epoch {epoch:4d}  D_loss: {d_avg:.4f}  G_loss: {g_avg:.4f}")

        if epoch in snap_epochs:
            z_snap = np.random.randn(500, z_dim)
            snapshots[epoch] = G.forward(z_snap).copy()

    return G, snapshots


def main():
    np.random.seed(0)
    t0 = time.time()

    real_data = make_gaussian_mixture(n=1000)
    std = real_data.std(axis=0)
    real_norm = real_data / std

    G, snapshots = train(real_norm, z_dim=8, epochs=2000, batch_size=64, lr=2e-4)

    elapsed = time.time() - t0
    print(f"Runtime: {elapsed:.1f}s")

    snap_epochs = sorted(snapshots.keys())
    fig, axes = plt.subplots(1, len(snap_epochs), figsize=(4 * len(snap_epochs), 4))

    for ax, ep in zip(axes, snap_epochs):
        gen = snapshots[ep] * std
        ax.scatter(real_data[:, 0], real_data[:, 1], s=8, alpha=0.3, label='real')
        ax.scatter(gen[:, 0], gen[:, 1], s=8, alpha=0.5, color='tab:red', label='fake')
        ax.set_title(f'Epoch {ep}')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.legend(markerscale=2, fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'gan_results.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
