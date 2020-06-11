# model parameters:
from functools import reduce

from torch import optim

import transport_graph as tg
import torch

import oracles
import dual_func_calculator as dfc
from prox_h import ProxH

from data_reader.data_reader import *

class Model:
    def __init__(self, transport_graph, total_od_flow=None, mu=0.25, rho=0.15):
        self.graph = tg.TransportGraph(transport_graph)
        self.graph_correspondences = transport_graph.graph_correspondences
        self.total_od_flow = total_od_flow
        self.mu = mu
        self.rho = rho
        self.t = torch.tensor(np.array(transport_graph.graph[['Free Flow Time']], dtype='float64').flatten(), requires_grad=True)

        self.phi_big_oracle = oracles.PhiBigOracle(self.graph, self.graph_correspondences)
        self.primal_dual_calculator = dfc.PrimalDualCalculator(self.phi_big_oracle,
                                                          self.graph.freeflow_times, self.graph.capacities,
                                                          mu = self.mu, rho = self.rho)

    def loss(self):
        return self.primal_dual_calculator.dual_func_value(self.t)

    def new_loss(self):
        return self.primal_dual_calculator.new_dual_func_value(self.t)

    def get_flows(self):
        if len(self.graph_correspondences) == 0:
            return
        flows = self.t.detach() - self.graph.freeflow_times
        base_v = next(iter(self.graph_correspondences))
        in_edges = flows[self.graph.in_edges(base_v - 1)].sum()
        out_edges = flows[self.graph.out_edges(base_v - 1)].sum()
        in_flows_sum = reduce(lambda x, y: x+y, [self.graph_correspondences[i][base_v] if base_v in self.graph_correspondences[i] else 0 for i in self.graph_correspondences])
        out_lows_sum = reduce(lambda x, y: x+y, [self.graph_correspondences[base_v][i] for i in self.graph_correspondences[base_v]])
        coef_t = (in_edges - out_edges) / (in_flows_sum - out_lows_sum)
        flows = flows / coef_t
        return flows


    def solve(self, num_iters=10):
        optimizer = optim.Adam([self.t], lr=0.01)
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward(retain_graph=True)
            optimizer.step()
            # print("new", loss)
            # print("old", self.loss())
            # print(self.t.sum())
            print(self.primal_dual_calculator.primal_func_value(self.get_flows()))

        flows = self.get_flows()
        for v in self.graph_correspondences.keys():
            in_edges = flows[self.graph.in_edges(v - 1)[0]].sum()
            out_edges = flows[self.graph.out_edges(v - 1)[0]].sum()
            in_flows_sum = reduce(lambda x, y: x+y, [self.graph_correspondences[i][v] if v in self.graph_correspondences[i] else 0 for i in self.graph_correspondences])
            out_lows_sum = reduce(lambda x, y: x+y, [self.graph_correspondences[v][i] for i in self.graph_correspondences[v]])
            print(in_edges - out_edges - in_flows_sum + out_lows_sum)
        # flows = self.get_flows()
        # print(flows)


data_reader = DataReader()
graph = data_reader.GetGraphStructure("data/Anaheim_net.tntp", [0, 1, 2, 4])[0]
data_reader.GetGraphCorrespondences("data/Anaheim_trips.tntp")
model = Model(data_reader)
model.solve()