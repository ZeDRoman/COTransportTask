# model parameters:
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
        self.t = torch.tensor(np.array(transport_graph.graph[['Free Flow Time']], dtype = 'float64').flatten(), requires_grad=True)

        phi_big_oracle = oracles.PhiBigOracle(self.graph, self.graph_correspondences)
        self.primal_dual_calculator = dfc.PrimalDualCalculator(phi_big_oracle,
                                                          self.graph.freeflow_times, self.graph.capacities,
                                                          mu = self.mu, rho = self.rho)

    def loss(self):
        return self.primal_dual_calculator.dual_func_value(self.t)

    def solve(self, num_iters=1000):
        optimizer = optim.Adam([self.t], lr=0.1)
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward(retain_graph=True)
            optimizer.step()
            print(loss)
        print(self.t)


data_reader = DataReader()
graph = data_reader.GetGraphStructure("data/Anaheim_net.tntp", [0, 1, 2, 4])[0]
data_reader.GetGraphCorrespondences("data/Anaheim_trips.tntp")
model = Model(data_reader)
model.solve()