import numpy as np
import torch
import time


class BaseOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')


class AutomaticOracle(BaseOracle):
    """
    Oracle for automatic calculations of function kGamma * \Psi (t)
    kGamma -> +0
    """
    def __init__(self, source, graph, source_correspondences):
        self.graph = graph
        self.source_index = source - 1
        self.corr_targets = np.array(list(source_correspondences.keys()), dtype='int64') - 1
        self.corr_values = torch.from_numpy(np.array(list(source_correspondences.values())))
        self.distances = None

    def func(self, t_parameter):
        self.update_shortest_paths(t_parameter)
        return - torch.matmul(self.distances, self.corr_values)

    def update_shortest_paths(self, t_parameter):
        self.distances, _ = self.graph.shortest_distances(self.source_index, self.corr_targets, t_parameter)


class PhiBigOracle(BaseOracle):
    def __init__(self, graph, correspondences):
        self.graph = graph
        self.correspondences = correspondences
        self.auto_oracles = []
        for source, source_correspondences in self.correspondences.items():
            self.auto_oracles.append(AutomaticOracle(source, self.graph, source_correspondences))
        self.work_time = 0.0

    def func(self, t_parameter):
        tic = time.time()
        result = None
        for auto_oracle in self.auto_oracles:
            v = auto_oracle.func(t_parameter)
            if result is None:
                result = v
            else:
                result += v
        self.work_time += time.time() - tic
        return result
