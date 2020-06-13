import numpy as np
import torch


class PhiBigOracleForSource():
    """
    Oracle for automatic calculations of function kGamma * \Psi (t)
    kGamma -> +0
    """
    def __init__(self, source, graph, source_correspondences):
        self.graph = graph
        self.source_index = source - 1
        self.corr_targets = np.array(list(source_correspondences.keys()), dtype='int64') - 1
        self.corr_values = torch.from_numpy(np.array(list(source_correspondences.values())))


    def func(self, t_parameter):
        distances, _ = self.graph.shortest_distances(self.source_index, self.corr_targets, t_parameter)
        return - torch.matmul(distances, self.corr_values)

    def new_func(self, t_parameter):
        val = None
        for i in range(self.corr_targets.shape[0]):
            path_tensor = self.graph.get_path_tensor(self.source_index, self.corr_targets[i])
            v = self.corr_values[i] * torch.log(torch.exp(-torch.matmul(path_tensor, t_parameter)).sum())
            if val is None:
                val = v
            else:
                val += v
        return val


class PhiBigOracle():
    def __init__(self, graph, correspondences):
        self.graph = graph
        self.correspondences = correspondences
        self.func_current = None
        self.auto_oracles = []
        for source, source_correspondences in self.correspondences.items():
            self.auto_oracles.append(PhiBigOracleForSource(source, self.graph, source_correspondences))

    def _reset(self, t_parameter):
        self.func_current = None
        for auto_oracle in self.auto_oracles:
            v = auto_oracle.func(t_parameter)
            if self.func_current is None:
                self.func_current = v
            else:
                self.func_current += v

    def func(self, t_parameter):
        self._reset(t_parameter)
        return self.func_current

    def new_func(self, t_parameter):
        result = None
        for auto_oracle in self.auto_oracles:
            v = auto_oracle.func(t_parameter)
            if result is None:
                result = v
            else:
                result += v
        return result
