"""
Score-Based Generative Model (SMLD) — pure NumPy.

Song & Ermon 2019 "Generative Modeling by Estimating Gradients of the Data Distribution."

Architecture
------------
  ScoreNetwork: MLP  [x(2), log_sigma(1)] → score(2)
    Linear(3→64) → GELU → Linear(64→128) → GELU → Linear(128→64) → GELU → Linear(64→2)

Training
--------
  Denoising Score Matching loss:
    L = E[sigma^2 || s_theta(x+noise, sigma) + noise/sigma^2 ||^2]
  - Data: 8-Gaussian 2D mixture
  - sigma: geometric sequence from 0.01 to 10.0, L=10 levels

Sampling
--------
  Annealed Langevin dynamics:
    For each sigma (descending): n_steps of
      x += step_size/2 * s_theta(x, sigma) + sqrt(step_size) * noise

Output
------
  score/score_based_results.png — score field quiver plot + samples
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# GELU activation (pure NumPy)
# ---------------------------------------------------------------------------

def gelu(x):
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(tanh_arg)
    dtanh = 1.0 - tanh_val ** 2
    inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * dtanh * inner_grad


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class Linear:
    def __init__(self, in_dim, out_dim, scale=None):
        if scale is None:
            scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        N = self._x.shape[0] if self._x.ndim > 1 else 1
        self.grads[0][...] = self._x.T @ dout
        self.grads[1][...] = dout.sum(axis=0)
        return dout @ self.W.T


# ---------------------------------------------------------------------------
# Score Network
# ---------------------------------------------------------------------------

class ScoreNetwork:
    """Simple MLP: [x(2), log_sigma(1)] → score(2)

    Architecture: Linear(3,64)→GELU→Linear(64,128)→GELU→Linear(128,64)→GELU→Linear(64,2)
    """

    def __init__(self):
        self.fc1 = Linear(3, 64)
        self.fc2 = Linear(64, 128)
        self.fc3 = Linear(128, 64)
        self.fc4 = Linear(64, 2, scale=0.01)
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        # Cache for backward
        self._cache = None

    def forward(self, x, log_sigma):
        """
        x         : (N, 2)
        log_sigma : (N, 1) or scalar
        Returns   : (N, 2) score estimates
        """
        N = x.shape[0]
        if np.isscalar(log_sigma):
            ls = np.full((N, 1), log_sigma)
        else:
            ls = np.broadcast_to(np.array(log_sigma).reshape(-1, 1), (N, 1))

        inp = np.concatenate([x, ls], axis=1)  # (N, 3)

        h1 = self.fc1.forward(inp)
        a1 = gelu(h1)
        h2 = self.fc2.forward(a1)
        a2 = gelu(h2)
        h3 = self.fc3.forward(a2)
        a3 = gelu(h3)
        out = self.fc4.forward(a3)

        self._cache = (inp, h1, a1, h2, a2, h3, a3)
        return out

    def backward(self, dout):
        inp, h1, a1, h2, a2, h3, a3 = self._cache
        d = self.fc4.backward(dout)
        d = d * gelu_grad(h3)
        d = self.fc3.backward(d)
        d = d * gelu_grad(h2)
        d = self.fc2.backward(d)
        d = d * gelu_grad(h1)
        d = self.fc1.backward(d)
        return d


# ---------------------------------------------------------------------------
# Data: 8-Gaussian mixture
# ---------------------------------------------------------------------------

def make_8gaussian(n=2000, seed=42):
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    radius = 3.0
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    n_each = n // 8
    data = []
    for c in centers:
        data.append(rng.randn(n_each, 2) * 0.3 + c)
    return np.vstack(data)


# ---------------------------------------------------------------------------
# Loss and Adam
# ---------------------------------------------------------------------------

def denoising_score_matching_loss(net, x0, sigma_list):
    """
    Denoising Score Matching loss:
      L = E[sigma^2 || s_theta(x+noise, sigma) + noise/sigma^2 ||^2]

    Parameters
    ----------
    net        : ScoreNetwork
    x0         : (N, 2) clean data
    sigma_list : array of sigma values

    Returns (loss, gradient already stored in net.grads)
    """
    N = x0.shape[0]
    # Sample a random sigma for each data point
    sigma_idx = np.random.randint(0, len(sigma_list), N)
    sigmas = sigma_list[sigma_idx]  # (N,)

    noise = np.random.randn(*x0.shape)  # (N, 2)
    x_noisy = x0 + sigmas[:, None] * noise  # (N, 2)

    log_sigma = np.log(sigmas[:, None])  # (N, 1)
    score_pred = net.forward(x_noisy, log_sigma)  # (N, 2)

    # Target score: -noise / sigma
    target = -noise / sigmas[:, None]  # (N, 2)

    # Weighted MSE: sigma^2 * ||s_theta - target||^2
    diff = score_pred - target  # (N, 2)
    weights = sigmas[:, None] ** 2  # (N, 1)
    loss = np.mean(weights * diff ** 2)

    # Gradient w.r.t. score_pred
    dout = 2.0 * weights * diff / (N * 2)
    net.backward(dout)

    return loss


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(net, data, sigma_list, n_steps=5000, batch_size=256, lr=1e-3):
    optimizer = Adam(lr=lr)
    N = data.shape[0]
    losses = []

    for step in range(1, n_steps + 1):
        # Zero gradients
        for g in net.grads:
            g[...] = 0.0

        idx = np.random.randint(0, N, batch_size)
        x0 = data[idx]
        loss = denoising_score_matching_loss(net, x0, sigma_list)
        optimizer.update(net.params, net.grads)
        losses.append(loss)

        if step % 1000 == 0:
            avg_loss = np.mean(losses[-500:])
            print(f"  Step {step:5d}  Loss = {avg_loss:.5f}")

    return losses


# ---------------------------------------------------------------------------
# Sampling: Annealed Langevin Dynamics
# ---------------------------------------------------------------------------

def annealed_langevin_sampling(net, sigma_list, n=500, n_steps=100, step_size=0.01):
    """
    Annealed Langevin dynamics sampling.

    For each sigma in sigma_list (descending order):
        n_steps of: x += step_size/2 * s_theta(x, sigma) + sqrt(step_size) * noise

    Parameters
    ----------
    net        : trained ScoreNetwork
    sigma_list : array of sigma values (will be sorted descending)
    n          : number of samples
    n_steps    : Langevin steps per noise level
    step_size  : step size for Langevin update

    Returns (n, 2) samples
    """
    x = np.random.randn(n, 2) * sigma_list.max()  # random init at highest noise
    sigmas_desc = np.sort(sigma_list)[::-1]

    for sigma in sigmas_desc:
        log_sig = np.full((n, 1), np.log(sigma))
        for _ in range(n_steps):
            score = net.forward(x, log_sig)
            noise = np.random.randn(*x.shape)
            x = x + step_size / 2.0 * score + np.sqrt(step_size) * noise

    return x


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_score_field(net, sigma, ax, xlim=(-5, 5), ylim=(-5, 5), grid_n=20):
    """Plot score field as quiver arrows at given sigma."""
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (G, 2)

    log_sig = np.log(sigma)
    score = net.forward(grid_pts, log_sig)  # (G, 2)

    U = score[:, 0].reshape(grid_n, grid_n)
    V = score[:, 1].reshape(grid_n, grid_n)
    # Clip magnitude for visualization
    mag = np.sqrt(U ** 2 + V ** 2 + 1e-8)
    U_norm = U / (mag + 1)
    V_norm = V / (mag + 1)

    ax.quiver(XX, YY, U_norm, V_norm, mag, cmap='plasma', alpha=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import time
    np.random.seed(0)
    t0 = time.time()

    # Data
    data = make_8gaussian(n=2000)
    print(f"Data shape: {data.shape}")
    print(f"Data range: x=[{data[:,0].min():.2f}, {data[:,0].max():.2f}], "
          f"y=[{data[:,1].min():.2f}, {data[:,1].max():.2f}]")

    # Sigma schedule: geometric from 0.01 to 10.0, L=10 levels
    L = 10
    sigma_list = np.geomspace(0.01, 10.0, L)
    print(f"Sigma levels: {sigma_list.round(4)}")

    # Build and train network
    net = ScoreNetwork()
    print(f"\n--- Training ScoreNetwork (5000 steps) ---")
    losses = train(net, data, sigma_list, n_steps=5000, batch_size=256, lr=3e-4)

    print(f"\nFinal avg loss: {np.mean(losses[-500:]):.5f}")
    print(f"Runtime: {time.time()-t0:.1f}s")

    # Generate samples via annealed Langevin sampling
    print("\n--- Annealed Langevin Sampling ---")
    samples = annealed_langevin_sampling(
        net, sigma_list, n=500, n_steps=100, step_size=1e-3
    )
    print(f"Generated samples: {samples.shape}")
    print(f"Sample range: x=[{samples[:,0].min():.2f}, {samples[:,0].max():.2f}], "
          f"y=[{samples[:,1].min():.2f}, {samples[:,1].max():.2f}]")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: score fields at different sigma levels
    xlim, ylim = (-5.5, 5.5), (-5.5, 5.5)

    for i, (ax, sigma_idx) in enumerate(zip(axes[0], [0, 4, 9])):
        sigma = sigma_list[sigma_idx]
        plot_score_field(net, sigma, ax, xlim=xlim, ylim=ylim, grid_n=18)
        ax.scatter(data[:, 0], data[:, 1], s=6, alpha=0.2, color='tab:blue', label='data')
        ax.set_title(f'Score field: σ={sigma:.3f}', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')

    # Bottom-left: real data
    axes[1, 0].scatter(data[:, 0], data[:, 1], s=8, alpha=0.4, color='tab:blue')
    axes[1, 0].set_title('Real data (8-Gaussian)', fontsize=10)
    axes[1, 0].set_xlim(*xlim)
    axes[1, 0].set_ylim(*ylim)
    axes[1, 0].set_aspect('equal')

    # Bottom-middle: generated samples
    axes[1, 1].scatter(data[:, 0], data[:, 1], s=6, alpha=0.15, color='tab:blue', label='real')
    axes[1, 1].scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.5,
                       color='tab:orange', label='generated')
    axes[1, 1].set_title('Generated samples (Langevin)', fontsize=10)
    axes[1, 1].set_xlim(*xlim)
    axes[1, 1].set_ylim(*ylim)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].legend(markerscale=2, fontsize=8)

    # Bottom-right: training loss curve
    axes[1, 2].plot(losses, alpha=0.4, linewidth=0.5, color='tab:gray')
    # Smooth
    window = 200
    smooth = np.convolve(losses, np.ones(window) / window, mode='valid')
    axes[1, 2].plot(np.arange(window - 1, len(losses)), smooth,
                    linewidth=1.5, color='tab:red', label='smoothed')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('DSM Loss')
    axes[1, 2].set_title('Training Loss (Denoising Score Matching)', fontsize=10)
    axes[1, 2].legend(fontsize=8)

    plt.suptitle('Score-Based Generative Model (SMLD)\n8-Gaussian 2D Mixture', fontsize=13)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'score_based_results.png')
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
