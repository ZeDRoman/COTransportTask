import numpy as np
import torch

class PrimalDualCalculator:
    def __init__(self, phi_big_oracle, freeflowtimes, capacities, rho = 10.0, mu = 0.25):
        self.links_number = len(freeflowtimes)
        self.rho = rho
        self.mu = mu
        self.freeflowtimes = torch.from_numpy(freeflowtimes) #\bar{t}
        self.capacities = torch.from_numpy(capacities)       #\bar{f}
        self.phi_big_oracle = phi_big_oracle

    def h_func(self, times):
        if self.mu == 0:
            h_function_value = torch.matmul(self.capacities, torch.max(times - self.freeflowtimes, 0))
        else:
            h_function_value = torch.sum(self.capacities * (torch.max(times - self.freeflowtimes, torch.zeros(times.shape, dtype=torch.double))) *
                                      (torch.max(times - self.freeflowtimes, torch.zeros(times.shape, dtype=torch.double)) /
                                       (self.rho * self.freeflowtimes)) ** self.mu) / (1.0 + self.mu)
        return h_function_value

    def dual_func_value(self, times):
        return self.phi_big_oracle.func(times)

    def new_dual_func_value(self, times):
        return self.phi_big_oracle.new_func(times)

    def primal_func_value(self, flows):
        if self.mu == 0:
            return np.dot(self.freeflowtimes, flows)  #sigma_sum_function
        else:
            return np.dot(self.freeflowtimes * flows,
                          self.rho * self.mu / (1.0 + self.mu) *
                          (flows / self.capacities) ** (1.0 / self.mu) + 1.0)  #sigma_sum_function

    def duality_gap(self, times, flows):
        return self.dual_func_value(times) + self.primal_func_value(flows)

    #for Frank-Wolfe algorithm
    def times_function(self, flows):
        return self.freeflowtimes * (1.0 + self.rho * np.power(flows / self.capacities, 1.0 / self.mu))
