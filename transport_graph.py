# Attention: as shown on the table above
# nodes indexed from 0 to ...
# edges indexed from 0 to ...
import graph_tool.all as gt
import graph_tool.topology as gtt
import numpy as np
import math

import torch
import os

SAVE_DIR = "torch_path/"

def get_tree_order(nodes_number, targets, pred_arr):
    #get nodes visiting order for flow calculation
    visited = np.zeros(nodes_number, dtype = np.bool_)
    sorted_vertices = [0] * 0
    for vertex in targets:
        temp = []
        while (not visited[vertex]):
            visited[vertex] = True
            if pred_arr[vertex] != vertex:
                temp.append(vertex)
                vertex = pred_arr[vertex]
        sorted_vertices[0:0] = temp
    return sorted_vertices

class TransportGraph:
    def __init__(self, graph_data, maxpath_const = 3):
        graph_table = graph_data.graph

        self.nodes_number = graph_data.nodes_number
        self.links_number = graph_data.links_number
        self.max_path_length = maxpath_const * int(math.sqrt(self.links_number))
        
        self.graph = gt.Graph(directed=True)
        # nodes indexed from 0 to V-1
        vlist = self.graph.add_vertex(self.nodes_number)
        # let's create some property maps
        ep_freeflow_time = self.graph.new_edge_property("double")
        ep_capacity = self.graph.new_edge_property("double")
        
        # define data for edge properties
        self.capacities = np.array(graph_table[['Capacity']], dtype = 'float64').flatten()
        self.freeflow_times = np.array(graph_table[['Free Flow Time']], dtype = 'float64').flatten()  

        # adding edges to the graph
        inits = np.array(graph_table[['Init node']], dtype = 'int64').flatten()
        terms = np.array(graph_table[['Term node']], dtype = 'int64').flatten()
        for index in range(self.links_number):
            init_index = graph_table['Init node'][index] - 1
            term_index = graph_table['Term node'][index] - 1
            edge = self.graph.add_edge(self.graph.vertex(init_index),
                                       self.graph.vertex(term_index))
            ep_freeflow_time[edge] = self.freeflow_times[index]
            ep_capacity[edge] = self.capacities[index]
            
        # save properties to graph
        self.graph.edge_properties["freeflow_times"] = ep_freeflow_time
        self.graph.edge_properties["capacities"] = ep_capacity
        
        self.pred_to_edge = [{}] * self.nodes_number
        for node_index in range(self.nodes_number):
            self.pred_to_edge[node_index] = {source: edge_index 
                                             for source, _, edge_index in self.in_edges(node_index)}

        self.path_tensors = {}
       
    def get_graphtool(self):
        return self.graph

    def successors(self, node_index):
        return self.graph.get_out_neighbors(node_index)

    def predecessors(self, node_index):
        return self.graph.get_in_neighbors(node_index)
        
    #source, target and index of an edge
    def in_edges(self, node_index):
        return self.graph.get_in_edges(node_index, [self.graph.edge_index])
    
    #source, target and index of an edge
    def out_edges(self, node_index):
        return self.graph.get_out_edges(node_index)
    
    def shortest_distances(self, source, targets, times):
        ep_time_map = self.graph.new_edge_property("double", np.maximum(times.detach().numpy(), self.freeflow_times))
        distances, pred_map = gtt.shortest_distance(g = self.graph,
                                                    source = source,
                                                    target = targets,
                                                    weights = ep_time_map,
                                                    pred_map = True)
        sorted_vertices = get_tree_order(self.nodes_number, targets, pred_map)
        pred_edges = [self.pred_to_edge[vertex][pred_map[vertex]] for vertex in sorted_vertices]
        path = {}
        for j in targets:
            i = j
            path[j] = []
            while i != source:
                i_next = pred_map[i]
                path[j].append(self.pred_to_edge[i][i_next])
                i = i_next
        distances = torch.stack([torch.index_select(times, 0, torch.tensor(path[i])).sum() for i in targets])
        return distances, pred_map.a

    def create_path_tensor(self, source, target, path_amount=100, path_length=30):
        file = SAVE_DIR + "{}_{}.pt".format(source, target)
        if os.path.isfile(file):
            return torch.load(file)
        path_generator = gtt.all_paths(self.graph, source, target, path_length)
        path_tensor = torch.zeros((path_amount, self.links_number), dtype=torch.double)
        for i in range(path_amount):
            p = next(path_generator)
            for j in range(1, len(p)):
                path_tensor[i][self.pred_to_edge[p[j]][p[j - 1]]] = 1
        torch.save(path_tensor, file)
        return path_tensor

    def get_path_tensor(self, source, target):
        if source not in self.path_tensors:
            self.path_tensors[source] = {}
        if target not in self.path_tensors[source]:
            self.path_tensors[source][target] = self.create_path_tensor(source, target)
        return self.path_tensors[source][target]