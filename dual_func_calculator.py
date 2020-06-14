import numpy as np
import torch


class PrimalDualCalculator:
    def __init__(self, phi_big_oracle, freeflow_times, capacities, rho=10.0, mu=0.25):
        self.links_number = len(freeflow_times)
        self.rho = rho
        self.mu = mu
        self.freeflow_times = torch.from_numpy(freeflow_times)
        self.capacities = torch.from_numpy(capacities)
        self.phi_big_oracle = phi_big_oracle

    # calculates h(t) = \Sigma \sigma_e^* (t_e)
    def h_func(self, times):
        delta_t = torch.max(times - self.freeflow_times, torch.zeros(times.shape, dtype=torch.double))
        if self.mu == 0:
            h_function_value = torch.matmul(self.capacities, delta_t)
        else:
            h_function_value = torch.sum(self.capacities *
                                         delta_t *
                                         (delta_t / (self.rho * self.freeflow_times)) ** self.mu) / (1.0 + self.mu)
        return h_function_value

    # calculates \Phi(t)
    def phi_big_value(self, times):
        return self.phi_big_oracle.func(times)

    # calculates \Sigma \sigma_e (f_e)
    def primal_func_value(self, flows):
        if self.mu == 0:
            return np.dot(self.freeflow_times, flows)
        else:
            return np.dot(self.freeflow_times * flows,
                          self.rho * self.mu / (1.0 + self.mu) *
                          (flows / self.capacities) ** (1.0 / self.mu) + 1.0)

    def duality_gap(self, times, flows):
        return self.h_func(times) + self.phi_big_value(times) + self.primal_func_value(flows)
