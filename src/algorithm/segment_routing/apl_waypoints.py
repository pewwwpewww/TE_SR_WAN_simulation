''' Idea is to optimize the MLU together with the weighted average path length by selecting waypoint in such a way that we try to minimalize an objective which is formed by combining the MLU and the weighted average path length '''
''' The objective is calculated using a lambda and (lambda*w_apl) + (1-lambda)*mlu for lambda = 0.5 we have an equal split for lambda < 0.5 we prioritize the mlu and for lambda > 0.5 analogically we prioritize w_apl'''
''' Base idea is to have a robust network with spread out utilization by factoring MLU but also have short paths leading to low latency by factoring the weighted average path length '''


import time

import networkit as nk
import numpy as np
from collections import defaultdict

from algorithm.generic_sr import GenericSR

class AplWaypoints(GenericSR):
    
    def __init__(self, nodes: list, links: list, demands: list, weights: dict = None, waypoints: dict = None, **kwargs):
        super().__init__(nodes, links, demands, weights, waypoints)
        self.__links = links #List of links with capacities {(u,v,c)...}
        self.__demands = demands #List of demands with src, dst, d
        self.__n = len(nodes)
        self.__weights = weights if weights else {(u, v): 1. for u, v,_ in self.__links}
        
        
        self.__dist = {}  #Dictonary of form {(u,v): dist(u->v)}
        self.__path =  {} #Dictonary of form (u,v) Path from u -> v
        
        #Initialize a nk Graph for our network
        self.__graph = None
        self.__init_graph()
    
        self.__candidate_waypoints = self.__select_k_waypoint_candidates()
        
        
        if self.__graph is not None:
            self.__init_bfs()   #Initializes a bfs for all relevant nodes thus fills __dist and __path
        
        self.__original_path_length = float("inf")
        self.__total_weighted_path_length = 0
        self.__total_demand_weight = 0
        self.__init_link_arrays()
    
    ''' Initializes a networkit graph to allow for BFS and other networkit functions later on '''
    def __init_graph(self):
        self.__graph = nk.Graph(weighted = False, directed = True, n = self.__n)    #Unweighted since we optimize 
        for u,v,_ in self.__links:
            self.__graph.addEdge(u,v)
        return
    
    ''' 
        Initializes the data structures used for keeping track and updating the flow and keeping the capacities 
        The link index map tells us an index for each link (u,v) which we can use to get the corresponding values out of the other data structures 
    '''
    def __init_link_arrays(self):
        self.__link_index_map = {}  #Map where we store a index for each link to access the flows array efficiently
        self.__num_links = len(self.__links)
        self.__flows = np.zeros(self.__num_links, dtype=float)  #array for the flows over each link access over link index map 
        self.__capacities = np.ones(self.__num_links, dtype=float)
        
        for i, (u,v,c) in enumerate(self.__links):
            self.__link_index_map[(u,v)] = i
            self.__capacities[i] = c
    
    ''' 
            Calculates the flows along the graph by iterating over all demands
            For each demand we take the path using the global __path variables (see __init_bfs) and then add the demand value d for all links along the calculated path
    '''
    def __calculate_flows(self):
        #reset the flow map
        self.__flows.fill(0.0)
        
        for s,t,d in self.__demands:
            if s == t:
                continue
            
            path = self.__path.get((s,t), [])
            
            if len(path) < 2:   #e.g the path doesnt exist
                continue
            for u, v in zip(path[:-1], path[1:]):
                l_idx = self.__link_index_map[(u,v)]
                if l_idx is not None:
                    self.__flows[l_idx] += d
                else:
                    print(f"Link ({u},{v}) not found in the index map")
        return self.__flows
    
    '''
        Updates the flow array according to a waypoint being set for a demand (s,t,d) by handling as if we have two demands one from s to waypoint and waypoint to t
        We reset the original path and add the flow to the new path in two segments
        
    '''
    def __update_flows(self, flow_array, s, t, d, waypoint):
        if s == t or waypoint is None or waypoint == s or waypoint == t:
            return flow_array
        old_path = self.__path.get((s,t), [])
        new_path = self.__path.get((s,waypoint), [])[:-1] + self.__path.get((waypoint, t), [])
        
        if len(new_path) < 2:
            return flow_array
        
        #Reset the old path
        for u,v in zip(old_path[:-1], old_path[1:]):
            l_idx = self.__link_index_map[(u,v)]
            if l_idx is not None:
                flow_array[l_idx] -= d
        
        #Go over the new path and add the flow
        for u,v in zip(new_path[:-1], new_path[1:]):
            l_idx = self.__link_index_map[(u,v)]
            if l_idx is not None:
                flow_array[l_idx] += d
        return flow_array
    
    '''
        Calculates the utilization as in the flow over each node divided by the capacitie 
        return an array accessible using self.__link_index_map and the mlu e.g. the maximum of the utilization array 
    '''
    def __calculate_utilization(self, flow_array):
        util_array = flow_array / self.__capacities
        mlu = np.max(util_array)
        return util_array, mlu
    
    '''
        Calculates objective with a weight lambda which controls how much the w_apl and the mlu account for
        For lambda = 0.5 --> w_apl and mlu are equally weighted
        For lambda > 0.5 --> w_apl is weighted more heavily than mlu in respect to how close lambda is to 1 (1 for only w_apl)
        For lambda < 0.5 --> mlu is weighted more heavily than w_apl in respect to how close lambda is to 0 (0 for only mlu)
    '''
    def __calculate_objective(self, w_apl, mlu, lmbd=0.3):
        ''' '''
        return (lmbd * w_apl) + (1 - lmbd) * mlu
    
    '''
        Initializes BFS to calculate all relevant paths and distances, a relevant path being either from s to t for all demands as well as from s to waypoint and waypoint to t for all demands and all waypoints
        Saves the distances in the dictonarys self.__dist and self.__path
    '''
    def __init_bfs(self):
        #Get all unique sources from the demands
        sources = set(u for u,_,_ in self.__demands)

        #Loop over all uniqe sources and calculate BFS
        for src in sources:
            #Get all shortest paths
            bfs = nk.distance.BFS(self.__graph, src, storePaths=True)
            bfs.run()
            distances = bfs.getDistances()  #Distances are hop counts since __graph is unweighted 
            
            for s,t,d in self.__demands:
                if src == s:
                    dist = distances[t]
                    
                    if dist < float("inf") and dist >= 1:
                        self.__dist[(s,t)] = dist
                        self.__path[(s,t)] = bfs.getPath(t)
                    else:
                        self.__dist[(s,t)] =  None
                        self.__path[(s,t)] = None
                    
                    for waypoint in self.__candidate_waypoints:
                        dist_w = bfs.distance(waypoint)
                    
                        if dist_w < float("inf") and dist_w >= 1:
                            self.__dist[(s,waypoint)] = dist_w
                            self.__path[(s,waypoint)] = bfs.getPath(waypoint)
                        else:
                            self.__dist[(s,waypoint)] = None
                            self.__path[(s,waypoint)] = None
            
        #Go over all paths and distances between waypoint to t
        targets = set(t for _,t,_ in self.__demands)
    
        for waypoint in self.__candidate_waypoints:
            bfs = nk.distance.BFS(self.__graph, waypoint, storePaths = True)
            bfs.run()
            distances = bfs.getDistances()
            for t in targets:
                dist = distances[t]
                
                if dist < float("inf") and dist >= 1: 
                    self.__dist[(waypoint,t)] = dist
                    self.__path[(waypoint,t)] = bfs.getPath(t)
                else:
                    self.__dist[(waypoint,t)] = None
                    self.__path[(waypoint,t)] = None
        
        return
    
    '''
        Selects a set of waypoints candidates which represent the betweenness centrality(how important a node is in respect to the connection to other nodes) and the
        demand volume (how important a node is in respect to its volume of demand e.g. demand hotspots)
        From each criteria we take the top k-nodes and combine the for the candidates e.g. the candidate set is at most 2*k large
    '''
    def __select_k_waypoint_candidates(self, k = 10):
        k = max(10, round(0.2*self.__n))
        
        #Calculate betweenness centrality (see formula) and normalize e.g. normalized in interval [0,1]
        #Option of using approximation implementation 
        bc = nk.centrality.Betweenness(self.__graph)
        bc.run()
        
        #bc scores (as np.array to allow array indexing with an np array on the top_k_bc)
        bc_scores = np.array(bc.scores())
        
        #Now select high demand nodes
        demand_volume = defaultdict(float)  
        
        #Go over the demands 
        for s,t,d in self.__demands:
            demand_volume[s] += d
            demand_volume[t] += d
        
        #demand volume is a dict with node:volume
        #Convert to np.arrays sort them and extract the top_k_demand_volume indices
        volume_values = np.array(list(demand_volume.values()))
        
        #Compute degree
        degree_score = [self.__graph.degree(v) for v in range(self.__n)]

        score = []
        for v in range(self.__n):
            s = bc_scores[v] * 0.5 + degree_score[v] * 0.3 + volume_values[v] * 0.15
            score.append(s)

        sorted_indices = np.argsort(score)[::-1]
        
        #return the top k scores
        return sorted_indices[:k]
    
    '''
        Calculates the weighted average path length using the distance between two nodes given by self.__dist
    '''
    def __calculate_weighted_apl(self):
        total__weighted_path_length = 0
        total_demand_weight = 0
        
        for s,t,d in self.__demands:
            if s != t:
                dist = self.__dist.get((s,t), float("inf"))
                if dist != float("inf") and dist >= 1:
                    total__weighted_path_length += dist * d
                    total_demand_weight += d
            else:
                continue
        
        self.__total_weighted_path_length = total__weighted_path_length
        self.__total_demand_weight = total_demand_weight
        
        return total__weighted_path_length / total_demand_weight if total_demand_weight > 0 else 0
    
    '''
        Updates the weighted apl by the change which occurs when we add a waypoint to a demand
        The change is defined by delta see formula below
        e.g old_apl + delta is the new apl
    '''
    def __update_weighted_apl(self, s, t, d, waypoint, old_objective):
        old_len = self.__dist.get((s,t))
        s_to_waypoint = self.__dist.get((s,waypoint))
        waypoint_to_t = self.__dist.get((waypoint,t))
        
        if old_len is None or s_to_waypoint is None or waypoint_to_t is None: 
            return old_objective
        
        new_len = s_to_waypoint + waypoint_to_t
        delta = ((new_len-old_len)*d) / self.__total_demand_weight 
        
        return old_objective + delta
    
    '''
        Creates a loads dictonary out of the util_array conforming to the specifications in generic_sr
    '''
    def __calculate_loads(self, util_array):
        '''Returns a dict in the form loads[(u,v)] = utilization on said link e.g. conforming to generic_sr.py '''
        loads = {}
        for u,v,_ in self.__links:
            loads[(u,v)] = util_array[self.__link_index_map[(u,v)]]
            
        return loads
    
    '''
        Goes over all waypoints for each demand to find the waypoints that give the best_objective (minimize)
    '''
    def __apl_optimization(self):
        #get candidate set, go over all demands(sorted) and optimize for each demand the waypoints 
        #Candidate set is initialized in __init__
        
        #Get wapl and mlu 
        self.__calculate_flows()
        best_util_array, best_mlu = self.__calculate_utilization(self.__flows)
        best_w_apl = self.__calculate_weighted_apl()
        
        #Combine for objective
        best_objective_apl = self.__calculate_objective(best_w_apl, best_mlu)
        
        #For updating the objective properly
        original_objective = best_objective_apl
        
        #Sort demands 
        original_indices = list(range(len(self.__demands)))
        sorted_indices = sorted(original_indices, key=lambda i: self.__demands[i][2], reverse = True)
        
        waypoints = dict()
        
        for d_idx in sorted_indices:
            best_waypoint = None
            s,t,d = self.__demands[d_idx]
            for waypoint in self.__candidate_waypoints:
                if waypoint == s or waypoint == t:
                    continue
                #Update our weighted apl
                w_apl = self.__update_weighted_apl(s, t, d, waypoint, original_objective)
                
                #update the flow and the mlu aswell as the util_array
                flow_array = self.__update_flows(self.__flows, s, t, d, waypoint)
                util_array, mlu = self.__calculate_utilization(flow_array)
                objective_apl = self.__calculate_objective(w_apl, mlu)
                
                if objective_apl < best_objective_apl:
                    best_objective_apl = objective_apl
                    best_waypoint = waypoint
                    best_util_array = util_array
                    best_flows = flow_array
                    
            if best_waypoint is not None:
                waypoints[d_idx] = [(s, best_waypoint), (best_waypoint, t)]
            else:
                waypoints[d_idx] = [(s,t)]
                
        loads =  self.__calculate_loads(best_util_array)
        
        return loads, waypoints, best_objective_apl
                
            
    '''
        Computes a solution for the problem and returns according to generic_sr
    '''
    def solve(self) -> dict:
        self.__start_time = t_start = time.time() #start timer sys wide
        pt_start = time.process_time() #start process timer
        
        #Execute process 
        loads, waypoints, objective_apl = self.__apl_optimization()
        
        pt_duration = time.process_time() - pt_start
        t_duration =time.time() - t_start
        
        #print(f"Waypoints: {waypoints}")
        
        solution = {
            "objective_mlu": max(loads.values()),
            "objective_alu": sum(loads.values()) / len(loads),
            "objective_apl": objective_apl,
            "execution_time": t_duration,
            "process_time": pt_duration,
            "waypoints": waypoints,
            "weights": self.__weights,
            "loads": loads,
        }
         
        return solution
         
    def get_name(self):
        """ returns name of algorithm """
        return f"apl_waypoints"
            
            
        
        
        
        
                    
