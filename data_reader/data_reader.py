from scanf import scanf
import re
import numpy as np
import pandas as pd


class DataReader:
    graph = None
    nodes_number = None
    links_number = None
    graph_correspondences = None
    total_od_flow = None
    values_dict = None

    # TODO One data reader func

    def GetGraphStructure(self, file_name, columns_order):
        """
            Reads graph from file, see Anaheim_net.tntp for example
            :param file_name: name of file containing graph saved in .tntp format
            :param columns_order: list of 4 column indexes:
                    first index - edge tail
                    second index - edge head
                    third index - edge capacity
                    forth index - edge length, a.k.a. Free flow time
                    other columns ignored
            :return: df - pandas dataframe representing graph
            :return edges_number - number of edges
        """

        if self.graph is not None:
            return self.graph, self.links_number

        with open(file_name, 'r') as file:
            data = file.read()

        # skip <NUMBER OF ZONES>, <FIRST THRU NODE>, <NUMBER OF NODES> parameters, will get from graph correspondences
        # reads some graph metadata
        nodes_number = scanf('<NUMBER OF NODES> %d', data)[0]
        links_number = scanf('<NUMBER OF LINKS> %d', data)[0]

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

        return graph, links_number

    def GetGraphCorrespondences(self, file_name):
        """
           Reads information about sources and destinations of flow, see Anaheim_trips.tntp for example
           Flow is transferred only between origins (first <NUMBER OF ZONES> nodes)
           :param file_name: name of file containing list of origins (sources), saved in .tntp format
            for each origin - list of flows to all the other origins
           :return: graph_correspondences - map from origin number. value - map[dstOriginId] -> flow.
                    Thus, map represents required amount of flow to be transfered between pair of origins.
           :return total_od_flow - total amount of flow
        """

        if self.graph_correspondences is not None:
            return self.graph_correspondences, self.total_od_flow

        with open(file_name, 'r') as file:
            trips_data = file.read()

        # skip <NUMBER OF ZONES> parameter, will get from size of graph_correspondences
        # reads some graph metadata
        total_od_flow = scanf('<TOTAL OD FLOW> %f', trips_data)[0]

        # splitting origins
        p = re.compile("Origin[ \t]+[\d]+")
        origins_list = p.findall(trips_data)
        origins = np.array([int(re.sub('[a-zA-Z ]', '', line)) for line in origins_list])

        p = re.compile("\n"
                       "[0-9.:; \n]+"
                       "\n\n")
        res_list = p.findall(trips_data)
        res_list = [re.sub('[\n \t]', '', line) for line in res_list]

        # origin to map
        graph_correspondences = {}
        for origin_index in range(0, len(origins)):
            origin_correspondences = res_list[origin_index].strip('[\n;]').split(';')
            graph_correspondences[origins[origin_index]] = \
                dict([scanf("%d:%f", line) for line in origin_correspondences])

        self.graph_correspondences = graph_correspondences
        self.total_od_flow = total_od_flow
        return graph_correspondences, total_od_flow

    def ReadAnswer(self, filename):
        """
        Reads best known answer for current transport network
        :param filename: name of file, saved in .tntp format
        :return: dict
                dict['flow'] - list of flow for each edge
                dict['time'] - TODO
        """

        if self.values_dict is not None:
            return self.values_dict

        with open(filename) as file:
            lines = file.readlines()

        datalist = []
        for line in lines:
            if re.search('[a-zA-Z]', line) is None and re.search('[0-9]', line) is not None:
                datalist.append(line)

        lines = [line.strip('[\t; ]') for line in datalist]

        values_dict = {'flow': [], 'time': []}

        for line in lines:
            items = line.split()
            values_dict['flow'].append(float(items[3]))
            values_dict['time'].append(float(items[4]))

        self.values_dict = values_dict
        return values_dict
