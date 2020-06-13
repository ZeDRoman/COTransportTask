from functools import reduce

import graph_tool.all as gt
import graph_tool.topology as gtt
import numpy as np

import torch


class TransportGraph:
    def __init__(self, data_reader):
        graph_table = data_reader.graph

        self.nodes_number = data_reader.nodes_number
        self.links_number = data_reader.links_number

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

        # target_edge -> source_edge -> edge
        self.pred_to_edge = [{}] * self.nodes_number
        for node_index in range(self.nodes_number):
            self.pred_to_edge[node_index] = {source: edge_index
                                             for source, _, edge_index in self.in_edges(node_index)}

    # source, target and index of an edge
    def in_edges(self, node_index):
        return self.graph.get_in_edges(node_index, [self.graph.edge_index])

    # source, target and index of an edge
    def out_edges(self, node_index):
        return self.graph.get_out_edges(node_index, [self.graph.edge_index])

    # get length of shortest distance from source to each target
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

    # calculates sum of in-flow and out-flow for one vertex
    def _get_in_out_flows_sum(self, flows, v):
        in_edges = list(map(lambda x: x[2], self.in_edges(v - 1)))
        out_edges = list(map(lambda x: x[2], self.out_edges(v - 1)))

        in_edges_sum = flows[in_edges].sum()
        out_edges_sum = flows[out_edges].sum()
        return in_edges_sum, out_edges_sum

    # calculates sum of original in-flow and out-flow for one vertex from data
    @staticmethod
    def _get_in_out_flows_sum_from_data(v, graph_correspondences):
        in_flows_sum = reduce(lambda x, y: x + y,
                              [graph_correspondences[i][v] if v in graph_correspondences[i] else 0
                               for i in
                               graph_correspondences])
        out_flows_sum = reduce(lambda x, y: x + y,
                               [graph_correspondences[v][i] for i in graph_correspondences[v]])
        return in_flows_sum, out_flows_sum

    # Normalize graph flows to original flow sizes
    def get_flows(self, t, graph_correspondences):
        if len(graph_correspondences) == 0:
            return
        flows = t
        base_v = next(iter(graph_correspondences))
        in_edges_sum, out_edges_sum = self._get_in_out_flows_sum(flows, base_v)
        in_flows_sum, out_flows_sum = self._get_in_out_flows_sum_from_data(base_v, graph_correspondences)

        coef_t = (in_edges_sum - out_edges_sum) / (in_flows_sum - out_flows_sum)
        flows = flows / coef_t
        return flows

    # Check graph is correct transport graph
    def check_graph(self, flows, graph_correspondences):
        for v in graph_correspondences.keys():
            in_edges_sum, out_edges_sum = self._get_in_out_flows_sum(flows, v)
            in_flows_sum, out_flows_sum = self._get_in_out_flows_sum_from_data(v, graph_correspondences)
            print(in_edges_sum - out_edges_sum - in_flows_sum + out_flows_sum)
