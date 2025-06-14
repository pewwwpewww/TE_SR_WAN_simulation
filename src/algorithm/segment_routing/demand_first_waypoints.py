"""
todo briefly describe
"""

import time

import networkit as nk
import numpy as np

from algorithm.generic_sr import GenericSR


class DemandsFirstWaypoints(GenericSR):
    BIG_M = 10 ** 9

    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)

        # topology info
        self.__capacities = self.__extract_capacity_dict(links)  # dict with {(u,v):c, ..}
        self.__links = list(self.__capacities.keys())  # list with [(u,v), ..]
        self.__n = len(nodes)
        self.__capacity_map = None

        # demand segmentation and aggregate to matrix
        # store all target nodes for Some pairs shortest path algorithm
        self.__demands = demands

        # initial weights
        self.__weights = weights if weights else {(u, v): 1. for u, v in self.__links}

        # networKit graph and some pairs shortest path (SPSP) algorithm
        self.__g = None
        self.__apsp = None

        self.__init_graph()
        self.__init_capacity_map()
        return

    @staticmethod
    def __extract_capacity_dict(links):
        """ Converts the list of link/capacities into a capacity dict (compatibility reasons)"""
        return {(u, v): c for u, v, c in links}

    def __init_capacity_map(self):
        self.__capacity_map = np.ones((self.__n, self.__n), np.float)
        for u, v in self.__links:
            self.__capacity_map[u][v] = self.__capacities[u, v]

    def __init_graph(self):
        """ Create networKit graph, add weighted edges and create spsp (some pairs shortest path) object """
        self.__g = nk.Graph(weighted=True, directed=True, n=self.__n)
        for u, v in self.__links:
            self.__g.addEdge(u, v, self.__weights[u, v])
        self.__apsp = nk.distance.APSP(self.__g)

    def __compute_distances(self):
        """ Recomputes the shortest path for 'some' pairs """
        self.__apsp.run()
        return self.__apsp.getDistances()

    def __get_shortest_path_fraction_map(self, distances):
        link_fraction_map = np.zeros((self.__n, self.__n, self.__n, self.__n), np.float)

        for s in range(self.__n):
            # iterate over nodes sorted by distance
            u_map = dict(zip(range(self.__n), np.array(distances[s]).argsort()))

            for t in range(self.__n):
                if s == t:
                    continue
                node_fractions = np.zeros(self.__n, np.float)
                node_fractions[s] = 1

                for u_idx in range(self.__n - 1):
                    u = u_map[u_idx]
                    fraction = node_fractions[u]
                    if not fraction:
                        continue

                    successors = list(v for v in self.__g.iterNeighbors(u) if
                                      self.__weights[(u, v)] == distances[u][t] - distances[v][t])

                    new_fraction = fraction / len(successors)
                    for v in successors:
                        link_fraction_map[s][t][u][v] = new_fraction
                        node_fractions[v] += new_fraction if v != t else 0.
        return link_fraction_map

    def __get_flow_map(self, sp_fraction_map):
        flow_map = np.zeros((self.__n, self.__n), np.float)
        for s, t, d in self.__demands:
            flow_map += sp_fraction_map[s][t] * d
        return flow_map

    def __compute_utilization(self, flow_map):
        util_map = (flow_map / self.__capacity_map)
        objective = np.max(util_map)
        return util_map, objective

    def __compute_average_utilization(self, flow_map):
        util_map = flow_map / self.__capacity_map
        average_utilization = np.mean(util_map)
        return util_map, average_utilization

    def __update_flow_map(self, sp_fraction_map, flow_map, s, t, d, waypoint):
        new_flow_map = flow_map - sp_fraction_map[s][t] * d
        new_flow_map += sp_fraction_map[s][waypoint] * d
        new_flow_map += sp_fraction_map[waypoint][t] * d
        return new_flow_map

    def calculate_apl(self, paths):
        """
            Compute the weighted average path length across all demands.
        """
        total_weight = 0
        total_hops = 0

        for idx, (s, t, d) in enumerate(self.__demands):
            path = paths.get(idx, [])
            if len(path) < 2:
                continue  # skip empty or invalid paths
            total_weight += d
            total_hops += (len(path) - 1) * d  # hops = len(path) - 1

        return total_hops / total_weight if total_weight > 0 else 0

    def rebuild_paths(self, waypoint_segments: dict) -> dict:
        """
        ## Build full hop-by-hop paths for each demand using Dijkstra(!!because here the graph is weighted) over each segment.
        """
        paths = {}

        for demand_id, segments in waypoint_segments.items():
            full_path = []

            for src, dst in segments:
                dijkstra = nk.distance.Dijkstra(self.__g, src, self.__weights)
                dijkstra.run()
                segment = dijkstra.getPath(dst)

                if not segment or len(segment) < 2 :
                    continue #skip empty segments

                # # merge segments carefully to avoid duplicate nodes
                if full_path and full_path[-1] == segment[0]:
                    full_path += segment[1:]
                else:
                    full_path += segment

            paths[demand_id] = full_path

        return paths
    
    def __demands_first_waypoints(self):
        """ main procedure """
        distances = self.__compute_distances()
        sp_fraction_map = self.__get_shortest_path_fraction_map(distances)
        best_flow_map = self.__get_flow_map(sp_fraction_map)
        best_util_map, best_objective_mlu = self.__compute_utilization(best_flow_map)

        waypoints = dict()
        sorted_demand_idx_map = dict(zip(range(len(self.__demands)), np.array(self.__demands)[:, 2].argsort()[::-1]))
        for d_map_idx in range(len(self.__demands)):
            d_idx = sorted_demand_idx_map[d_map_idx]
            s, t, d = self.__demands[d_idx]
            best_waypoint = None
            for waypoint in range(self.__n):
                if waypoint == s or waypoint == t:
                    continue
                flow_map = self.__update_flow_map(sp_fraction_map, best_flow_map, s, t, d, waypoint)
                util_map, objective_mlu = self.__compute_utilization(flow_map)
                util_map, objective_alu = self.__compute_average_utilization(flow_map)

                if objective_mlu < best_objective_mlu:
                    best_flow_map = flow_map
                    best_util_map = util_map
                    best_objective_mlu = objective_mlu
                    best_waypoint = waypoint

            if best_waypoint is not None:
                waypoints[d_idx] = [(s, best_waypoint), (best_waypoint, t)]
            else:
                waypoints[d_idx] = [(s, t)]
        loads = {(u, v): best_util_map[u][v] for u, v, in self.__links}
        return loads, waypoints, best_objective_mlu, objective_alu

    def solve(self) -> dict:
        """ compute solution """

        self.__start_time = t_start = time.time()  # sys wide time
        pt_start = time.process_time()  # count process time (e.g. sleep excluded and count per core)
        loads, waypoints, objective_mlu, objective_alu = self.__demands_first_waypoints()
        pt_duration = time.process_time() - pt_start
        t_duration = time.time() - t_start

        paths = self.rebuild_paths(waypoints)
        
        solution = {
            "objective_mlu": objective_mlu,
            "objective_alu": objective_alu,
            "objective_apl": self.calculate_apl(paths),
            "execution_time": t_duration,
            "process_time": pt_duration,
            "waypoints": waypoints,
            "weights": self.__weights,
            "loads": loads
        }
        return solution

    def get_name(self):
        """ returns name of algorithm """
        return f"demand_first_waypoints"
