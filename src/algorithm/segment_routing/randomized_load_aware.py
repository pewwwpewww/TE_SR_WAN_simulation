import time
import random
import networkx as nx
from itertools import islice

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.sr_utility import SRUtility


class RandomizedLoadAwarePathSelection(GenericSR):
    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)
        self.__nodes = nodes
        self.__capacities = self.__extract_capacity_dict(links)
        self.__links = list(self.__capacities.keys())  # Ensures correct format
        self.__K = 3

        if waypoints is not None:
            segmented_demands = SRUtility.get_segmented_demands(waypoints, demands)
            self.__demands = {idx: (s, t, d) for idx, (s, t, d) in enumerate(segmented_demands)}
        else:
            self.__demands = {idx: (s, t, d) for idx, (s, t, d) in enumerate(demands)}

        self.__weights = weights if weights else {link: 1 for link in self.__links}

    @staticmethod
    def __extract_capacity_dict(links):
        capacity_dict = {}
        for link in links:
            if len(link) == 3:
                u, v, c = link
            elif len(link) == 2:
                u, v = link
                c = 1
            else:
                raise ValueError(f"Invalid link format: {link}")
            capacity_dict[(u, v)] = c
        return capacity_dict

    def __build_graph(self):
        G = nx.DiGraph()
        for (u, v), c in self.__capacities.items():
            G.add_edge(u, v, capacity=c, load=0, weight=int(self.__weights.get((u, v), 1)))
        return G

    def __k_shortest_paths(self, G, s, t, k):
        try:
            return list(islice(nx.shortest_simple_paths(G, s, t, weight="weight"), k))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def __path_score(self, path, demand, current_flow_map):
        score = 0
        for u, v in zip(path[:-1], path[1:]):
            load = current_flow_map.get((u, v), 0) + demand
            capacity = self.__capacities[(u, v)]
            util = load / capacity
            score += util
        return score

    def __create_flow_map(self, paths: dict) -> dict:
        """
        Create a map showing the flow on each link.
        """
        flow_map = {link: 0 for link in self.__links}
        for idx, path in paths.items():
            _, _, demand = self.__demands[idx]
            for u, v in zip(path[:-1], path[1:]):
                flow_map[(u, v)] += demand
        return flow_map

    def __compute_utilization_and_loads(self, G):
        """
        Compute link loads and utilization based on flow map.
        """
        flow_map = self.__create_flow_map({idx: [u for u, v in zip(path[:-1], path[1:])] + [path[-1]]
                                           for idx, path in self.__paths.items()})
        util_map = {}
        loads = {}
        max_util = 0

        for (u, v), capacity in self.__capacities.items():
            load = flow_map.get((u, v), 0)
            util = load / capacity if capacity > 0 else 0
            util_map[(u, v)] = util
            loads[(u, v)] = load
            max_util = max(max_util, util)

        self.__flow_map = flow_map  # Store for later use
        return loads, util_map, max_util

    def __calculate_average_link_utilization(self, flow_map: dict) -> float:
        """
        Calculate average utilization across all links.
        """
        total_utilization = 0
        for (u, v), flow in flow_map.items():
            capacity = self.__capacities.get((u, v), 1)
            total_utilization += flow / capacity if capacity > 0 else 0
        return total_utilization / len(flow_map)

    def solve(self) -> dict:
        self.__flow_map = {link: 0 for link in self.__links}
        wall_start = time.time()
        cpu_start = time.process_time()

        G = self.__build_graph()
        waypoints = {}
        paths = {}

        for idx, (s, t, d) in self.__demands.items():
            candidates = self.__k_shortest_paths(G, s, t, self.__K)
            best_score = float("inf")
            best_path = None

            for path in candidates:
                score = self.__path_score(path, d, self.__flow_map)
                if score < best_score:
                    best_score = score
                    best_path = path

            if best_path:
                for u, v in zip(best_path[:-1], best_path[1:]):
                    self.__flow_map[(u, v)] += d
                    G[u][v]["load"] += d

                paths[idx] = best_path
                if len(best_path) == 2:
                    waypoints[idx] = [(best_path[0], best_path[1])]
                else:
                    segs = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
                    waypoints[idx] = segs

        self.__paths = paths
        loads, _, objective_mlu = self.__compute_utilization_and_loads(G)
        objective_alu = self.__calculate_average_link_utilization(self.__flow_map)
        wall_time = time.time() - wall_start
        cpu_time = time.process_time() - cpu_start

        return {
            "objective_mlu": objective_mlu,
            "objective_alu": objective_alu,
            "objective_apl": -1,
            "execution_time": wall_time,
            "process_time": cpu_time,
            "waypoints": waypoints,
            "weights": {k: int(v) for k, v in self.__weights.items()},
            "loads": loads
        }

    def get_name(self):
        return "randomized_load_aware"