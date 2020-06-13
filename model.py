from functools import reduce
import torch

import dual_func_calculator as dfc

import oracles
import transport_graph as tg



class Model:
    def __init__(self, transport_graph, total_od_flow=None, mu=0.25, rho=0.15):
        self.graph = tg.TransportGraph(transport_graph)
        self.graph_correspondences = transport_graph.graph_correspondences
        self.total_od_flow = total_od_flow
        self.mu = mu
        self.rho = rho

        self.t = torch.tensor(self.graph.freeflow_times, requires_grad=True)
        self.grad_sum = None
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

    # calculates sum of in-flow and out-flow for one vertex
    def _get_in_out_flows_sum(self, flows, v):
        in_edges = list(map(lambda x: self.graph.pred_to_edge[x[1]][x[0]], self.graph.in_edges(
            v - 1)))
        out_edges = list(map(lambda x: self.graph.pred_to_edge[x[1]][x[0]], self.graph.out_edges(
            v - 1)))

        in_edges_sum = flows[in_edges].sum()
        out_edges_sum = flows[out_edges].sum()
        return in_edges_sum, out_edges_sum

    # calculates sum of in-flow and out-flow for one vertex from data
    def _get_in_out_flows_sum_from_data(self, v):
        in_flows_sum = reduce(lambda x, y: x + y,

                              [self.graph_correspondences[i][v] if v in self.graph_correspondences[i] else 0

                               for i in
                               self.graph_correspondences])
        out_flows_sum = reduce(lambda x, y: x + y,
                               [self.graph_correspondences[v][i] for i in self.graph_correspondences[v]])
        return in_flows_sum, out_flows_sum

    def get_flows(self, t):
        if len(self.graph_correspondences) == 0:
            return
        flows = t
        base_v = next(iter(self.graph_correspondences))
        in_edges_sum, out_edges_sum = self._get_in_out_flows_sum(flows, base_v)
        in_flows_sum, out_flows_sum = self._get_in_out_flows_sum_from_data(base_v)

        coef_t = (in_edges_sum - out_edges_sum) / (in_flows_sum - out_flows_sum)
        flows = flows / coef_t
        return flows

    def check_graph(self, flows):
        for v in self.graph_correspondences.keys():
            in_edges_sum, out_edges_sum = self._get_in_out_flows_sum(flows, v)
            in_flows_sum, out_flows_sum = self._get_in_out_flows_sum_from_data(v)
            print(in_edges_sum - out_edges_sum - in_flows_sum + out_flows_sum)


    def solve(self, optimizer, num_iters=1000, loss_history=False, verbose=False):
        """
        If loss_history is True, list of primal function values is returned
        :param optimizer:
        :param num_iters:
        :param loss_history:
        :param verbose: logging
        :return: flows, loss_history? TODO
        """
        primal_values = []

        for i in range(num_iters):
            optimizer.zero_grad()

            loss = self.loss_phi_big()
            loss.backward()

            if self.grad_sum is None:
                self.grad_sum = -self.t.grad
            else:
                self.grad_sum -= self.t.grad

            loss = self.loss_h()
            loss.backward()


            optimizer.step()

            primal_value = self.primal_dual_calculator.primal_func_value(self.get_flows(self.grad_sum))
            if verbose:
                print(primal_value)
            if loss_history:
                primal_values.append(primal_value)

        flows = self.get_flows(self.grad_sum)

        if loss_history:
            return flows, primal_values
        return flows


        flows = self.get_flows(self.grad_sum)

        if loss_history:
            return flows, primal_values
        return flows
      
data_reader = DataReader()
data_reader.read_graph("data/Anaheim_net.tntp", [0, 1, 2, 4])
data_reader.read_correspondences("data/Anaheim_trips.tntp")
model = Model(data_reader)
optimizer = torch.optim.SGD(params=[model.t], lr=0.0000001)
model.solve(optimizer, verbose=True)
