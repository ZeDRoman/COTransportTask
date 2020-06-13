import torch

from data_reader.data_reader import DataReader
from model import Model

data_reader = DataReader()
data_reader.read_graph("data/Anaheim_net.tntp", [0, 1, 2, 4])
data_reader.read_correspondences("data/Anaheim_trips.tntp")
model = Model(data_reader)
optimizer = torch.optim.SGD(params=[model.t], lr=0.0000001)
model.solve(optimizer, verbose=True)