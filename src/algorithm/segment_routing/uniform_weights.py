import time

from algorithm.generic_sr import GenericSR
from algorithm.segment_routing.equal_split_shortest_path import EqualSplitShortestPath


class UniformWeights(GenericSR):
    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)

        self.__nodes = nodes  # [i, ...]
        self.__links = links  # [(i,j,c), ...]
        self.__demands = demands  # {idx: (s,t,d), ...}
        self.__weights = {(i, j): 1 for i, j, c in self.__links}
        self.__waypoints = waypoints

    def calculate_apl(self, paths: dict) -> float:
        """##Calculates the Average Path Length (APL), weighted by demand volume."""
        total_weighted_length = 0
        total_demand = 0

        for idx, (s, t, d) in enumerate(self.__demands):
            path = paths[idx]
            if len(path) >= 2:
                total_weighted_length += (len(path) - 1) * d
                total_demand += d

        return total_weighted_length / total_demand if total_demand > 0 else 0

    def solve(self) -> dict:
        """ set weights to inverse capacity and use shortest path algorithm """

        # add random waypoint for each demand
        t = time.process_time()
        pt_start = time.process_time()  # count process time (e.g. sleep excluded)

        post_processing = EqualSplitShortestPath(nodes=self.__nodes, links=self.__links, demands=self.__demands,
                                                 split=True, weights=self.__weights, waypoints=self.__waypoints)
        solution = post_processing.solve()

        pt_duration = time.process_time() - pt_start
        exe_time = time.process_time() - t

        solution["objective_apl"] = self.calculate_apl(solution["paths"])

        # update execution time
        solution["execution_time"] = exe_time
        solution["process_time"] = pt_duration
        return solution

    def get_name(self):
        """ returns name of algorithm """
        return f"uniform_weights"
