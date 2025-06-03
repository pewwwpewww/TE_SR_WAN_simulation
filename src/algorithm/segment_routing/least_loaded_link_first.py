"""
todo briefly describe
"""

import time

import networkit as nk
import networkx as nx
import numpy as np

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.sr_utility import SRUtility


class LeastLoadedLinkFirst(GenericSR):
    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None, **kwargs): # type: ignore
        super().__init__(nodes, links, demands, weights, waypoints)

        # topology info
        self.__capacities = self.__extract_capacity_dict(links)  # dict with {(u,v):c, ..}
        self.__links = list(self.__capacities.keys())  # list with [(u,v), ..]
        self.__n = len(nodes)
        if waypoints is not None:
            segmented_demands = SRUtility.get_segmented_demands(waypoints, demands)
            self.__demands = {idx: (s, t, d) for idx, (s, t, d) in enumerate(segmented_demands)}  # dict {idx:(s,t,d)}
            self.__segments = waypoints  # dict with {idx:(p,q)}
        else:
            self.__demands = {idx: (s, t, d) for idx, (s, t, d) in enumerate(demands)}
            self.__segments = {idx: [(p, q)] for idx, (p, q, _) in enumerate(demands)}

        # initial weights
        self.__weights = weights if weights else {(u, v): 1. for u, v in self.__links}
        self.__flow_sum = {(u, v): 0 for u, v in self.__links}
        self.__utilization = {(u, v): 0 for u, v in self.__links}

        # networkX graph algorithm
        # self.__g = None

        self.__init_graph()
        self.__init_capacity_map()
        return

    @staticmethod
    def __extract_capacity_dict(links):
        """ Converts the list of link/capacities into a capacity dict (compatibility reasons)"""
        return {(u, v): c for u, v, c in links}

    def __init_capacity_map(self):
        self.__capacity_map = np.ones((self.__n, self.__n), dtype='f')
        for u, v in self.__links:
            self.__capacity_map[u][v] = self.__capacities[u, v]

    def __init_graph(self):
        """ Create networKit graph, add weighted edges and create spsp (some pairs shortest path) object """
        self.__g = nx.DiGraph()
        for n in range(self.__n):
            self.__g.add_node(n)
        for u, v in self.__links:
            self.__g.add_edge(u, v)
            self.__g.edges[u,v]["weight"] = self.__weights[u, v]

    def __potential_utilisation_dijkstra(self, s, t, demand):
        # This contains the utilisation for all nodes for their best paths from s
        node_utilisation = {u: float("inf") for u in list(self.__g)}
        # This contains the distance for all nodes for their best paths from s
        node_distance = {u: float("inf") for u in list(self.__g)}

        # This contains the predecessor of the node
        predecessor = {u: -1 for u in list(self.__g)}

        # This contains whether a node was already visited
        visited = {u: False for u in list(self.__g)}

        # The utilisation and distance from the start node to itself is of course 0
        node_utilisation[s] = 0
        node_distance[s] = 0

        # While there are nodes left to visit and t was not reached
        while visited[t] != True:

            # find all the nodes with the currently least utilisation from the start node
            minimum_utilisation = float("inf")
            possible_shortest_nodes = list()
            for u in list(self.__g):
                # ... by going through all nodes that haven't been visited yet
                if node_utilisation[u] < minimum_utilisation and not visited[u]:
                    minimum_utilisation = node_utilisation[u]
                    possible_shortest_nodes.clear()
                    possible_shortest_nodes.append(u)
                elif node_utilisation[u] == minimum_utilisation and not visited[u]:
                    possible_shortest_nodes.append(u)

            if len(possible_shortest_nodes) == 0:
                # There was no node not yet visited --> We are done
                return None
            
            # get the node with the shortest path
            shortest_node = min(possible_shortest_nodes, key=lambda u: node_distance[u])

            # then, for all neighboring nodes that haven't been visited yet
            neighbours = list(self.__g.neighbors(shortest_node))
            for n in neighbours:
                # if the path over this edge has a lower utilisation
                if node_utilisation[n] > max(node_utilisation[shortest_node], (self.__flow_sum[shortest_node, n] + demand)/self.__capacities[shortest_node, n]):
                    # Save this path as new best path
                    predecessor[n] = shortest_node
                    node_distance[n] = node_distance[shortest_node]+1
                    node_utilisation[n] = max(node_utilisation[shortest_node], (self.__flow_sum[shortest_node, n] + demand)/self.__capacities[shortest_node, n])

            # Lastly, note that we are finished with this node.
            visited[shortest_node] = True

        if predecessor[t] == -1:
            # Node t was not reached
            return None

        # Generates path
        path = list()
        path.append(t)
        while(predecessor[path[0]] != -1):
            path.insert(0, predecessor[path[0]])
        # print(f"s: {s}")
        # print(f"t: {t}")
        # print(f"path: {path}")
        # print(f"predecessor: {predecessor}")
        return path

    def __least_loaded_link_first(self):
        """ main procedure """
        
        # sorts the demand list decending
        sorted_demands = sorted(self.__demands.items(), key=lambda item: -item[1][2])

        # list for storing all the routes for the demands
        routes = list()

        # print(f"graph: {self.__g}")
        for idx, [s, t, demand] in sorted_demands:
            # print(f"s: {s}, t: {t}, demand: {demand}")

            # gets a good path with minimal link utilization with dijkstra
            best_path = self.__potential_utilisation_dijkstra(s, t, demand)
            if(best_path is None):
                # There was an Error with the path
                print(f"Could not find a Path for this demand: s={s}, t={t}, demand={demand}")
                raise Exception(f"Could not find a Path for this demand: s={s}, t={t}, demand={demand}")
            
            # Adds the demand to that path
            # print(f"best_path: {best_path}")
            for i in range(len(best_path)-1):
                self.__flow_sum[best_path[i], best_path[i+1]] += demand
                
            routes.append(best_path)

    def solve(self) -> dict:
        """ compute solution """

        self.__start_time = t_start = time.time()  # sys wide time
        pt_start = time.process_time()  # count process time (e.g. sleep excluded and count per core)
        self.__least_loaded_link_first()
        pt_duration = time.process_time() - pt_start
        t_duration = time.time() - t_start

        utilization = {(i, j): self.__flow_sum[i, j] / self.__capacities[i, j] for i, j in self.__links}
        avg_util = np.mean(list(utilization.values()))

        solution = {
            "objective_mlu": max(utilization.values()),
            "objective_alu": sum(utilization.values()) / len(utilization),
            "objective_apl": -1,
            "execution_time": t_duration,
            "process_time": pt_duration,
            "waypoints": self.__segments,
            "weights": self.__weights,
            "loads": utilization,
            "avg_util": avg_util
        }

        return solution

    def get_name(self):
        """ returns name of algorithm """
        return f"least_loaded_link_first"
