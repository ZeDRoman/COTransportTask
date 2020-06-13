import transport_graph as tg
import dual_func_calculator as dfc
import oracles

import torch


class Model:
    def __init__(self, data_reader, mu=0.25, rho=0.15):
        self.graph = tg.TransportGraph(data_reader)
        self.graph_correspondences = data_reader.graph_correspondences
        self.mu = mu
        self.rho = rho

        self.t = torch.tensor(self.graph.freeflow_times, requires_grad=True)
        self.grad_sum = torch.zeros(self.t.size())
        self.phi_big_oracle = oracles.PhiBigOracle(self.graph, self.graph_correspondences)
        self.primal_dual_calculator = dfc.PrimalDualCalculator(self.phi_big_oracle,
                                                               self.graph.freeflow_times,
                                                               self.graph.capacities,
                                                               mu=self.mu,
                                                               rho=self.rho)

    def loss_phi_big(self):
        return self.primal_dual_calculator.phi_big_value(self.t)

    def loss_h(self):
        return self.primal_dual_calculator.h_func(self.t)

    # main logic here
    def solve(self, optimizer, num_iters=1000, loss_history=False, verbose=False):
        """
        If loss_history is True, list of primal function values is returned
        :param optimizer: torch optimizer
        :param num_iters: number of iterations
        :param loss_history: flag if loss history to be returned
        :param verbose: logging
        :return: flows - flow for edge in resulting solution
        :return: loss_history - history of primary func value change
        """
        primal_values = []

        for i in range(num_iters):
            optimizer.zero_grad()

            loss = self.loss_phi_big()
            loss.backward()

            self.grad_sum -= self.t.grad

            loss = self.loss_h()
            loss.backward()

            optimizer.step()

            if verbose or loss_history:
                primal_value = self.primal_dual_calculator.primal_func_value(
                    self.graph.get_flows(self.grad_sum, self.graph_correspondences))
            if verbose:
                print(primal_value)
            if loss_history:
                primal_values.append(primal_value)

        flows = self.graph.get_flows(self.grad_sum, self.graph_correspondences)

        if loss_history:
            return flows, primal_values
        return flows
