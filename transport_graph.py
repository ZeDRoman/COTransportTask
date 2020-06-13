# Attention: as shown on the table above
# nodes indexed from 0 to ...
# edges indexed from 0 to ...
import graph_tool.all as gt
import graph_tool.topology as gtt
import numpy as np
import math

import torch

class TransportGraph:
    def __init__(self, graph_data):
        graph_table = graph_data.graph

        self.nodes_number = graph_data.nodes_number
        self.links_number = graph_data.links_number
        
        self.graph = gt.Graph(directed=True)
        # nodes indexed from 0 to V-1
        self.graph.add_vertex(self.nodes_number)
        # adding edges to the graph
        for index in range(self.links_number):
            init_index = graph_table['Init node'][index] - 1
            term_index = graph_table['Term node'][index] - 1
            self.graph.add_edge(self.graph.vertex(init_index),
                                self.graph.vertex(term_index))

        # define data for edge properties
        self.capacities = np.array(graph_table[['Capacity']], dtype='float64').flatten()
        self.freeflow_times = np.array(graph_table[['Free Flow Time']], dtype='float64').flatten()
        
        self.pred_to_edge = [{}] * self.nodes_number
        for node_index in range(self.nodes_number):
            self.pred_to_edge[node_index] = {source: edge_index 
                                             for source, _, edge_index in self.in_edges(node_index)}
        
    #source, target and index of an edge
    def in_edges(self, node_index):
        return self.graph.get_in_edges(node_index, [self.graph.edge_index])
    
    #source, target and index of an edge
    def out_edges(self, node_index):
        return self.graph.get_out_edges(node_index, [self.graph.edge_index])
    
    def shortest_distances(self, source, targets, times):
        ep_time_map = self.graph.new_edge_property("double", np.maximum(times.detach().numpy(), self.freeflow_times))
        distances, pred_map = gtt.shortest_distance(g=self.graph,
                                                    source=source,
                                                    target=targets,
                                                    weights=ep_time_map,
                                                    pred_map=True)
        path = {}
        for j in targets:
            i = j
            path[j] = []
            while i != source:
                i_next = pred_map[i]
                path[j].append(self.pred_to_edge[i][i_next])
                i = i_next
        distances = torch.stack([torch.index_select(times, 0, torch.tensor(path[i])).sum() for i in targets])
        return distances
