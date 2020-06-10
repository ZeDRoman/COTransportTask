from data_reader.data_reader import DataReader
from model import Model
import torch
data_reader = DataReader()

dict = data_reader.ReadAnswer("data/Anaheim_flow.tntp")

graph = data_reader.GetGraphStructure("data/Anaheim_net.tntp", [0, 1, 2, 4])[0]
data_reader.GetGraphCorrespondences("data/Anaheim_trips.tntp")
model = Model(data_reader)
print(torch.tensor(dict['time'], dtype=torch.double).sum())