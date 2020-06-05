import numpy as np
from numba import njit

@njit
def newton(x_0_arr, a_arr, mu,
           tol = 1e-7, max_iter = 1000):
    """
    Newton method for equation x - x_0 + a x^mu = 0, x >= 0
    """
    res = np.empty(len(x_0_arr), dtype = np.float_)
    for i in range(len(x_0_arr)):
        x_0 = x_0_arr[i]
        a = a_arr[i]
        if x_0 <= 0:
            res[i] = 0
            continue
        x = min(x_0, (x_0 / a) ** (1 / mu))
        for it in range(max_iter):
            x_next = x - f(x, x_0, a, mu) / der_f(x, x_0, a, mu)
            if x_next <= 0:
                x_next = 0.1 * x
            x = x_next
            if np.abs(f(x, x_0, a, mu)) < tol:
                break
        res[i] = x
    return res

@njit
def f(x, x_0, a, mu):
    return x - x_0 + a * x ** mu

@njit
def der_f(x, x_0, a, mu):
    return 1.0 + a * mu * x ** (mu - 1)

class ProxH:
    def __init__(self, freeflowtimes, capacities, rho = 10.0, mu = 0.25):  
        self.links_number = len(freeflowtimes)
        self.rho_value = rho
        self.mu_value = mu
        self.freeflowtimes = freeflowtimes
        self.capacities = capacities

    def __call__(self, point, A, u_start = None):
        #print('argmin called...' + 'u_start = ' + str(u_start))
        if self.mu_value == 0:
            return np.maximum(point - A * self.capacities, self.freeflowtimes)
        self.A = A
        if u_start is None:
            u_start = 2.0 * self.freeflowtimes
        x = newton(x_0_arr = (point - self.freeflowtimes) / (self.rho_value * self.freeflowtimes),
                   a_arr = A * self.capacities / (self.rho_value * self.freeflowtimes),
                   mu = self.mu_value)
        argmin = (1 + self.rho_value * x) * self.freeflowtimes
        #print('my result argmin = ' + str(argmin))
        return argmin
