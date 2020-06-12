from functools import reduce
import torch
from torch import optim

from data_reader.data_reader import *
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

    def get_flows(self, t):
        if len(self.graph_correspondences) == 0:
            return
        flows = t
        base_v = next(iter(self.graph_correspondences))

        in_edges = list(map(lambda x: self.graph.pred_to_edge[x[1]][x[0]], self.graph.in_edges(
            base_v - 1)))
        out_edges = list(map(lambda x: self.graph.pred_to_edge[x[1]][x[0]], self.graph.out_edges(
            base_v - 1)))

        in_edges_sum = flows[in_edges].sum()
        out_edges_sum = flows[out_edges].sum()
        in_flows_sum = reduce(lambda x, y: x + y,
                              [self.graph_correspondences[i][base_v] if base_v in self.graph_correspondences[i] else 0 for i in
                               self.graph_correspondences])
        out_lows_sum = reduce(lambda x, y: x + y,
                              [self.graph_correspondences[base_v][i] for i in self.graph_correspondences[base_v]])

        coef_t = (in_edges_sum - out_edges_sum) / (in_flows_sum - out_lows_sum)
        flows = flows / coef_t
        return flows

    def check_graph(self, flows):
        for v in self.graph_correspondences.keys():
            in_edges = list(map(lambda x: self.graph.pred_to_edge[x[1]][x[0]], self.graph.in_edges(
                v - 1)))
            out_edges = list(map(lambda x: self.graph.pred_to_edge[x[1]][x[0]], self.graph.out_edges(
                v - 1)))

            in_edges_sum = flows[in_edges].sum()
            out_edges_sum = flows[out_edges].sum()
            in_flows_sum = reduce(lambda x, y: x + y,
                                  [self.graph_correspondences[i][v] if v in self.graph_correspondences[i] else 0 for i
                                   in
                                   self.graph_correspondences])
            out_lows_sum = reduce(lambda x, y: x + y,
                                  [self.graph_correspondences[v][i] for i in self.graph_correspondences[v]])

            print(in_edges_sum - out_edges_sum - in_flows_sum + out_lows_sum)

    def solve(self, num_iters=1000):
        optimizer = optim.RMSprop([self.t], lr=0.01)
        grad_sum = None
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = self.loss_phi_big()
            loss.backward()
            if grad_sum is None:
                grad_sum = -self.t.grad
            else:
                grad_sum -= self.t.grad
            loss = self.loss_h()
            loss.backward()
            optimizer.step()
            print(self.primal_dual_calculator.primal_func_value(self.get_flows(grad_sum)))

        flows = self.get_flows(grad_sum)

        #self.check_graph(flows)


data_reader = DataReader()
graph = data_reader.GetGraphStructure("data/Anaheim_net.tntp", [0, 1, 2, 4])[0]
data_reader.GetGraphCorrespondences("data/Anaheim_trips.tntp")
model = Model(data_reader)
model.solve()
