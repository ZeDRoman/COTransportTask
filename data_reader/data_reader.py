from typing import Dict, Any, List

from pandas import DataFrame
from scanf import scanf
import re
import numpy as np
import pandas as pd


class DataReader:
    def __init__(self):
        self.graph = None
        self.graph_correspondences = None
        self.nodes_number = None
        self.links_number = None
        self.flow = None

    def read_all(self, graph_filename=None, correspondences_filename=None, answer_filename=None):
        """
        Wrapper for reading all the data (where filename was provided)
        No return value, method fills class fields
        """
        if graph_filename is not None:
            self.read_graph(graph_filename)

        if correspondences_filename is not None:
            self.read_correspondences(correspondences_filename)

        if answer_filename is not None:
            self.read_answer(answer_filename)

    def read_graph(self, file_name, columns_order=(0, 1, 2, 4)):
        """
            Reads graph from file, see Anaheim_net.tntp for example
            :param file_name: name of file containing graph saved in .tntp format
            :param columns_order: list of 4 column indexes:
                    first index - edge tail
                    second index - edge head
                    third index - edge capacity
                    forth index - edge free flow time
                    other columns ignored
            self.graph - pandas dataframe with 4 given columns
            self.links_number - number of edges in graph
            self.nodes_number - number of vertices in graph
        """

        if self.graph is not None:
            return

        with open(file_name, 'r') as file:
            data = file.read()

        # skip <NUMBER OF ZONES>, <FIRST THRU NODE>, <NUMBER OF NODES> parameters,
        # will get necessary params from graph correspondences
        # reads some graph metadata
        nodes_number = scanf('<NUMBER OF NODES> %d', data)[0]
        links_number = scanf('<NUMBER OF LINKS> %d', data)[0]

        # column names
        headlist = ['Init node', 'Term node', 'Capacity', 'Free Flow Time']

        # get line list
        datalist = re.compile("[\t0-9.]+\t;").findall(data)

        # drop extra symbols in line end
        datalist = [line.strip('[\t;]') for line in datalist]

        # split line by tab
        datalist = [line.split('\t') for line in datalist]

        # construct pandas dataframe from provided column indexes
        graph = pd.DataFrame(np.asarray(datalist)[:, columns_order], columns=headlist)

        # edge tail, convert to integer
        graph['Init node'] = pd.to_numeric(graph['Init node'], downcast='integer')
        # edge head
        graph['Term node'] = pd.to_numeric(graph['Term node'], downcast='integer')

        # capacity
        graph['Capacity'] = pd.to_numeric(graph['Capacity'], downcast='float')
        # free flow time
        graph['Free Flow Time'] = pd.to_numeric(graph['Free Flow Time'], downcast='float')

        self.graph = graph
        self.links_number = links_number
        self.nodes_number = nodes_number

    def read_correspondences(self, file_name):
        """
            Reads information about sources and destinations of flow, see Anaheim_trips.tntp for example
            Flow is transferred only between origins (first <NUMBER OF ZONES> nodes usually, indexed from 1)
            :param file_name: name of file containing list of origins (sources), saved in .tntp format
            for each origin - list of flows to all the other origins
            self.graph_correspondences - map from origin number.
            value - map[dstOriginId] -> flow
            Thus, map represents required amount of flow to be transferred between pair of origins.
        """

        if self.graph_correspondences is not None:
            return

        with open(file_name, 'r') as file:
            trips_data = file.read()

        # splitting origins
        p = re.compile("Origin[ \t]+[\d]+")
        origins_list = p.findall(trips_data)
        origins = np.array([int(re.sub('[a-zA-Z ]', '', line)) for line in origins_list])

        # get origins correspondences
        p = re.compile("\n"
                       "[0-9.:; \n]+"
                       "\n\n")
        res_list = p.findall(trips_data)
        res_list = [re.sub('[\n \t]', '', line) for line in res_list]

        # origin to map
        graph_correspondences = {}
        for origin_index in range(0, len(origins)):
            # parse and save origins
            origin_correspondences = res_list[origin_index].strip('[\n;]').split(';')

            graph_correspondences[origins[origin_index]] = \
                dict([scanf("%d:%f", line) for line in origin_correspondences])

        self.graph_correspondences = graph_correspondences

    def read_answer(self, filename):
        """
        Reads best known answer for current transport network
        :param filename: name of file, saved in .tntp format,
        self.flow - flow for each edge in graph
        """

        if self.flow is not None:
            return

        with open(filename) as file:
            lines = file.readlines()

        datalist = []
        for line in lines:
            if re.search('[a-zA-Z]', line) is None and re.search('[0-9]', line) is not None:
                datalist.append(line)

        lines = [line.strip('[\t; ]') for line in datalist]

        flow = []

        for line in lines:
            items = line.split()
            flow.append(float(items[3]))

        self.flow = flow
