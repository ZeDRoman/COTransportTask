# model parameters:
import transport_graph as tg

import oracles
import dual_func_calculator as dfc
from prox_h import ProxH

class Model:
    def __init__(self, graph_data, graph_correspondences, total_od_flow, mu = 0.25, rho = 0.15):
        self.graph = tg.TransportGraph(graph_data)
        self.graph_correspondences = graph_correspondences
        self.total_od_flow = total_od_flow
        self.mu = mu
        self.rho = rho

    def find_equilibrium(self, solver_name = 'ustf', solver_kwargs = {}, verbose = False):

        prox_h = ProxH(self.graph.freeflow_times, self.graph.capacities, mu = self.mu, rho = self.rho)
        phi_big_oracle = oracles.PhiBigOracle(self.graph, self.graph_correspondences)
        primal_dual_calculator = dfc.PrimalDualCalculator(phi_big_oracle,
                                                          self.graph.freeflow_times, self.graph.capacities,
                                                          mu = self.mu, rho = self.rho)
        if verbose:
            print('Oracles created...')
            print(starting_msg)

        return result
