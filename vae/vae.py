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
    labels = []
    for i, c in enumerate(centers):
        pts = rng.randn(n_each, 2) * 0.4 + c
        data.append(pts)
        labels.extend([i] * n_each)
    return np.vstack(data), np.array(labels)


class Encoder:
    def __init__(self, latent_dim):
        W1 = he_init(2, 64)
        W2 = he_init(64, 32)
        W_mu = he_init(32, latent_dim)
        W_lv = he_init(32, latent_dim)
        b1, b2 = np.zeros(64), np.zeros(32)
        b_mu, b_lv = np.zeros(latent_dim), np.zeros(latent_dim)

        self.fc1 = Affine(W1, b1)
        self.relu1 = Relu()
        self.fc2 = Affine(W2, b2)
        self.relu2 = Relu()
        self.fc_mu = Affine(W_mu, b_mu)
        self.fc_lv = Affine(W_lv, b_lv)

        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2]

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        mu = self.fc_mu.forward(h)
        logvar = self.fc_lv.forward(h)
        return mu, logvar

    def backward(self, dmu, dlogvar):
        dh = self.fc_mu.backward(dmu) + self.fc_lv.backward(dlogvar)
        for layer in reversed(self.layers):
            dh = layer.backward(dh)
        return dh

    def params_and_grads(self):
        all_params, all_grads = [], []
        for layer in self.layers + [self.fc_mu, self.fc_lv]:
            all_params += layer.params
            all_grads += layer.grads
        return all_params, all_grads


class Decoder:
    def __init__(self, latent_dim):
        W1 = he_init(latent_dim, 32)
        W2 = he_init(32, 64)
        W3 = he_init(64, 2)
        b1, b2, b3 = np.zeros(32), np.zeros(64), np.zeros(2)

        self.fc1 = Affine(W1, b1)
        self.relu1 = Relu()
        self.fc2 = Affine(W2, b2)
        self.relu2 = Relu()
        self.fc3 = Affine(W3, b3)
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

    def params_and_grads(self):
        all_params, all_grads = [], []
        for layer in self.layers:
            all_params += layer.params
            all_grads += layer.grads
        return all_params, all_grads


class VAE:
    def __init__(self, latent_dim=2, beta=1.0, lr=1e-3):
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        enc_params, enc_grads = self.encoder.params_and_grads()
        dec_params, dec_grads = self.decoder.params_and_grads()
        self.params = enc_params + dec_params
        self.grads = enc_grads + dec_grads
        self.optimizer = Adam(lr=lr)

        self._eps = None

    def forward(self, x):
        mu, logvar = self.encoder.forward(x)
        self._eps = np.random.randn(*mu.shape)
        z = mu + self._eps * np.exp(0.5 * logvar)
        x_recon = self.decoder.forward(z)
        return x_recon, mu, logvar, z

    def loss(self, x, x_recon, mu, logvar):
        N = x.shape[0]
        recon = np.sum((x - x_recon) ** 2) / N
        kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar)) / N
        return recon + self.beta * kl, recon, kl

    def backward(self, x, x_recon, mu, logvar, z):
        N = x.shape[0]
        dx_recon = -2 * (x - x_recon) / N
        dz = self.decoder.backward(dx_recon)

        dmu_kl = self.beta * mu / N
        dlogvar_kl = self.beta * 0.5 * (np.exp(logvar) - 1) / N

        dmu_recon = dz * self._eps * 0.5 * np.exp(0.5 * logvar)
        dlogvar_recon = dz * self._eps * mu * 0.5 * np.exp(0.5 * logvar)

        # dz/dmu = 1, dz/dlogvar = 0.5 * exp(0.5*logvar) * eps
        dmu = dmu_kl + dz
        dlogvar = dlogvar_kl + dz * self._eps * 0.5 * np.exp(0.5 * logvar)

        self.encoder.backward(dmu, dlogvar)

    def step(self, x):
        x_recon, mu, logvar, z = self.forward(x)
        total, recon, kl = self.loss(x, x_recon, mu, logvar)
        self.backward(x, x_recon, mu, logvar, z)
        enc_params, enc_grads = self.encoder.params_and_grads()
        dec_params, dec_grads = self.decoder.params_and_grads()
        self.optimizer.update(enc_params + dec_params, enc_grads + dec_grads)
        return total, recon, kl

    def encode(self, x):
        mu, logvar = self.encoder.forward(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder.forward(z)

    def sample(self, n):
        z = np.random.randn(n, self.latent_dim)
        return self.decode(z)


def train(model, data, epochs=500, batch_size=64):
    N = data.shape[0]
    losses = []
    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            xb = data[idx[i:i + batch_size]]
            loss, _, _ = model.step(xb)
            total_loss += loss
            n_batches += 1
        losses.append(total_loss / n_batches)
    return losses


def main():
    np.random.seed(0)
    t0 = time.time()

    data, labels = make_gaussian_mixture(n=1000)

    model = VAE(latent_dim=2, beta=1.0, lr=1e-3)
    losses = train(model, data, epochs=500, batch_size=64)

    elapsed = time.time() - t0
    print(f"Final ELBO loss: {losses[-1]:.4f}  (runtime: {elapsed:.1f}s)")

    mu_all, _ = model.encode(data)

    generated = model.sample(200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for k in range(4):
        mask = labels == k
        axes[0].scatter(mu_all[mask, 0], mu_all[mask, 1], s=10,
                        alpha=0.6, color=colors[k], label=f'cluster {k}')
    axes[0].set_title('Latent Space')
    axes[0].legend(markerscale=2, fontsize=8)
    axes[0].set_aspect('equal')

    axes[1].scatter(data[:, 0], data[:, 1], s=10, alpha=0.4, label='real')
    axes[1].scatter(generated[:, 0], generated[:, 1], s=10, alpha=0.6,
                    color='tab:red', label='generated')
    axes[1].set_title('Generated vs Real')
    axes[1].legend(markerscale=2, fontsize=8)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'vae_results.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
