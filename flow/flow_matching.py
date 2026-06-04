import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_8gaussians(n=2000):
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    r = 3.0
    centers = np.c_[r*np.cos(angles), r*np.sin(angles)]
    idx = np.random.randint(8, size=n)
    return centers[idx] + np.random.randn(n,2)*0.3, idx

def time_emb(t, d=16):
    freqs = np.exp(-np.arange(d//2)*np.log(10000)/(d//2))
    e = t[:,None]*freqs[None,:]
    return np.concatenate([np.sin(e), np.cos(e)], axis=1)

class VelocityNet:
    def __init__(self, d=2, hidden=64, t_dim=16):
        self.t_dim = t_dim
        din = d + t_dim
        s = np.sqrt(2/(din+hidden))
        self.W1 = np.random.randn(din, hidden)*s
        self.b1 = np.zeros(hidden)
        s2 = np.sqrt(2/(hidden+hidden))
        self.W2 = np.random.randn(hidden, hidden)*s2
        self.b2 = np.zeros(hidden)
        s3 = np.sqrt(2/(hidden+d))
        self.W3 = np.random.randn(hidden, d)*s3
        self.b3 = np.zeros(d)
        self.params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
        self.grads  = [np.zeros_like(p) for p in self.params]

    def forward(self, x, t):
        te = time_emb(t, self.t_dim)
        h = np.concatenate([x, te], axis=1)
        h1 = np.maximum(h@self.W1+self.b1, 0)
        h2 = np.maximum(h1@self.W2+self.b2, 0)
        return h2@self.W3+self.b3

    def step(self, lr=1e-3, clip=1.0):
        for p,g in zip(self.params, self.grads):
            p -= lr*np.clip(g, -clip, clip)
            g[...] = 0

def train(net, data, epochs=2000, batch=256, lr=3e-4):
    N = len(data)
    losses = []
    for ep in range(epochs):
        idx = np.random.choice(N, batch)
        x1 = data[idx]
        x0 = np.random.randn(*x1.shape)
        t  = np.random.rand(batch)
        xt = (1-t[:,None])*x0 + t[:,None]*x1
        target = x1 - x0
        pred = net.forward(xt, t)
        loss = np.mean((pred-target)**2)
        # simple finite-diff grad approx for small batch
        eps = 1e-4
        for i,(p,g) in enumerate(zip(net.params, net.grads)):
            flat = p.ravel()
            for j in range(min(50, len(flat))):
                flat[j] += eps
                lp = np.mean((net.forward(xt,t)-target)**2)
                flat[j] -= 2*eps
                lm = np.mean((net.forward(xt,t)-target)**2)
                flat[j] += eps
                g.ravel()[j] = (lp-lm)/(2*eps)
        net.step(lr)
        losses.append(loss)
        if (ep+1)%500==0:
            print(f"  epoch {ep+1}/{epochs}  loss={loss:.4f}")
    return losses

def sample(net, n=500, steps=20):
    x = np.random.randn(n, 2)
    ts = np.linspace(0, 1, steps+1)
    for i in range(steps):
        t = np.full(n, ts[i])
        v = net.forward(x, t)
        x = x + v*(ts[i+1]-ts[i])
    return x

if __name__ == "__main__":
    np.random.seed(42)
    data, _ = make_8gaussians(3000)
    net = VelocityNet()
    print("Training Flow Matching...")
    losses = train(net, data, epochs=2000)
    samples = sample(net, 500)

    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].scatter(data[:,0], data[:,1], s=5, alpha=0.4); axes[0].set_title("Data")
    axes[1].scatter(samples[:,0], samples[:,1], s=5, c='orange', alpha=0.6); axes[1].set_title("Generated")
    axes[2].plot(losses[::50]); axes[2].set_title("Training Loss"); axes[2].set_xlabel("epoch×50")
    plt.tight_layout()
    plt.savefig("flow/flow_matching_results.png", dpi=100)
    print(f"Saved flow/flow_matching_results.png  final_loss={losses[-1]:.4f}")
