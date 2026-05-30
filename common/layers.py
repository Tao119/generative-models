import numpy as np


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x = x
        return x @ W + b

    def backward(self, dout):
        W, b = self.params
        dx = dout @ W.T
        dW = self.x.T @ dout
        db = dout.sum(axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Relu:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class LeakyRelu:
    def __init__(self, alpha=0.2):
        self.params, self.grads = [], []
        self.alpha = alpha
        self.x = None

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        return dout * np.where(self.x > 0, 1.0, self.alpha)


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        self.out = out
        return out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class Tanh:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        return dout * (1 - self.out ** 2)


class BatchNorm1D:
    def __init__(self, D, eps=1e-5, momentum=0.9):
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(D)
        self.running_var = np.ones(D)
        self.training = True
        self.cache = None

    def forward(self, x):
        if self.training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            self.cache = (x, x_hat, mu, var)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        x, x_hat, mu, var = self.cache
        N = x.shape[0]
        dgamma = (dout * x_hat).sum(axis=0)
        dbeta = dout.sum(axis=0)
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        dx_hat = dout * self.gamma
        std_inv = 1 / np.sqrt(var + self.eps)
        dx = (1 / N) * std_inv * (
            N * dx_hat
            - dx_hat.sum(axis=0)
            - x_hat * (dx_hat * x_hat).sum(axis=0)
        )
        return dx


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr_t = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g ** 2
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)


def he_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
