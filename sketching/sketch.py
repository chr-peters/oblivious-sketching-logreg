import math

import numpy as np
from numba import jit


@jit(nopython=True)
def _insert(X_prime, x_vec, alpha, b, N):
    x = np.random.randint(0, alpha)
    l = math.floor(np.log(x * (b - 1) + 1) / np.log(b))  # noqa: E741
    x = np.random.randint(0, N)
    elem = l * (N) + x
    X_prime[elem] = X_prime[elem] + x_vec

    return X_prime


class Sketch:
    def __init__(self, h_max, b, N, n, d):
        self.h_max = h_max
        self.b = b
        self.N = N
        self.n = n
        self.d = d

        shape = (N * (h_max + 1), d)
        self.X_prime = np.zeros(shape)

        self.alpha = 0
        for l in range(0, h_max + 1):  # noqa: E741
            self.alpha += b ** (l)

        self.p = np.repeat(0.0, h_max + 1)
        self.w = np.repeat(0.0, h_max + 1)
        for l in range(0, h_max + 1):  # noqa: E741
            self.p[l] = 1 / (self.alpha * (b ** (-l)))
            self.w[l] = 1 / self.p[l]

        self.weights = np.repeat(0.0, N * (h_max + 1))
        for l in range(0, N * (h_max + 1)):  # noqa: E741
            self.weights[l] = self.w[int(l / N)]

    def insert(self, x_vec):
        self.X_prime = _insert(self.X_prime, x_vec, self.alpha, self.b, self.N)

    def get_reduced_matrix(self):
        return self.X_prime
