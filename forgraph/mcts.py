import os   # added line
import os.path as osp   # added line
import config  # added line
import copy  # added line
import math
import timeit  # added line
import random   # added line
from typing import List, Tuple, Dict # added line
import torch
import numpy as np # added line
import networkx as nx
from Configures import mcts_args
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from shapley import mc_shapley, l_shapley, mc_l_shapley, gnn_score, NC_mc_l_shapley # added ,mc_l_shapley
from functools import partial
from collections import Counter
from pipeline import MCTSNode
from textwrap import wrap  # added line
from functools import partial # added line
from collections import Counter  # added line
from collections import defaultdict  # added line

# global xcounter   # added line
global ycounter    # added line
global BaseNum     # added line
global BaseNodes    # added line






class MCTSNode():

    def __init__(self, coalition: list, data: Data,
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.data = data
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


class MCTS():
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, n_rollout: int,
                 min_atoms: int, c_puct: float, expand_atoms: int, score_func):
        """ graph is a networkX graph """
        self.X = X
        self.edge_index = edge_index
        self.data = Data(x=self.X, edge_index=self.edge_index)
        self.graph = to_networkx(self.data, to_undirected=True)
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms

        self.root_coalition = sorted([i for i in range(self.num_nodes)])
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct)
        self.root = self.MCTSNodeClass(self.root_coalition)
        self.state_map = {str(self.root.coalition): self.root}

    def mcts_rollout(self, tree_node):


        # added these
        # ***************************************************
        
        def split_node_list_ordered(ordered_nodesA, BaseNum):
            try:
                n = BaseNum 
                first_nodes = []
                
                for item in ordered_nodesA[:n]:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        node = item[0]  # Just take the node, ignore degree if problematic
                        first_nodes.append(node)
                    else:
                        # Assume it's already a node identifier
                        first_nodes.append(item)
                
                second_nodes = [node for node in ordered_nodesA if node not in first_nodes]
                return first_nodes, second_nodes
                
            except Exception as e:
                print(f"Error in split_node_list_ordered: {e}")
                print(f"Input: {ordered_nodesA}")
                return [], ordered_nodesA

      
        def get_adjacent_nodes_and_remaining(root, edges, all_nodesx):
                """
                Returns the adjacent nodes of a root node and the remaining node set.
                
                Args:
                    root: The root node to find adjacent nodes for
                    edges: List of edges as tuples (node1, node2)
                    all_nodesx: Set of all nodes in the connected subgraph
                
                Returns:
                    tuple: (adjacent_nodes, remaining_nodes)
                """

                # # Convert root from tensor to integer if needed
                # if hasattr(root, 'item'):  # Check if it's a tensor
                #     root = root.item()
                # elif isinstance(root, (list, tuple)) and len(root) > 0:
                #     root = root[0]  # If it's a list/tuple with one element


                # Build adjacency list
                graph = defaultdict(set)
                for u, v in edges:
                    graph[u].add(v)
                    graph[v].add(u)
                
                # Get adjacent nodes of root
                adjacent_nodes = graph.get(root, set())
                
                # Calculate remaining nodes (all nodes except root)
                remaining_nodes = all_nodesx - {root}
                remaining_nodes = remaining_nodes - adjacent_nodes
                
                return adjacent_nodes, remaining_nodes
        

      
        def calculate_ordered_adjacent_nodes_betweenness(Adjacent_NodesToRoot_degree_Orderlist, graph, root_node):
                """
                Calculate betweenness centrality for nodes adjacent to the root node and order them
                based on the average of their degree and betweenness centrality.
                
                Parameters:
                Adjacent_NodesToRoot_degree_Orderlist (list): Ordered list of nodes adjacent to root node
                graph: NetworkX graph object containing the network
                root_node (int): The root node (default: 9)
                
                Returns:
                tuple: (ordered_nodes, betweenness_dict, combined_scores)
                    - ordered_nodes: List of nodes ordered by average of degree and betweenness centrality
                    - betweenness_dict: Dictionary with all adjacent nodes and their betweenness centrality
                    - combined_scores: Dictionary with combined scores for each node
                """

                # # Convert root from tensor to integer if needed
                # if hasattr(root_node, 'item'):  # Check if it's a tensor
                #     root_node = root_node.item()
                # elif isinstance(root_node, (list, tuple)) and len(root_node) > 0:
                #     root_node = root_node[0]  # If it's a list/tuple with one element

                
                # Validate input
                if not Adjacent_NodesToRoot_degree_Orderlist:
                    return [], {}, {}
                
                if root_node not in graph.nodes():
                    raise ValueError(f"Root node {root_node} not found in graph")
                
                # Calculate betweenness centrality for all nodes
                betweenness = nx.betweenness_centrality(graph, normalized=True)
                
                # Get degree for all nodes
                degrees = dict(graph.degree())
                
                # Normalize degree values to 0-1 range for fair averaging with betweenness
                max_degree = max(degrees.values()) if degrees.values() else 1
                normalized_degrees = {node: deg / max_degree for node, deg in degrees.items()}
                
                # Calculate combined score (average of normalized degree and betweenness)
                combined_scores = {}
                for node in Adjacent_NodesToRoot_degree_Orderlist:
                    norm_degree = normalized_degrees.get(node, 0)
                    betw_centrality = betweenness.get(node, 0)
                    combined_scores[node] = (norm_degree + betw_centrality) / 2
                
                # Order nodes by combined score (descending order)
                ordered_nodesA = sorted(Adjacent_NodesToRoot_degree_Orderlist, 
                                    key=lambda x: combined_scores[x], 
                                    reverse=True)
                
                # Create betweenness dictionary for only adjacent nodes
                adjacent_betweenness = {node: betweenness.get(node, 0.0) for node in Adjacent_NodesToRoot_degree_Orderlist}
                
                return ordered_nodesA, adjacent_betweenness, combined_scores
        





        def calculate_ordered_non_adjacent_nodes_betweenness(graph, root_node):
                """
                Calculate betweenness centrality for nodes NOT adjacent to the root node and order them
                based on the average of their degree and betweenness centrality.
                
                Parameters:
                graph: NetworkX graph object containing the network
                root_node (int): The root node (default: 9)
                
                Returns:
                tuple: (ordered_nodes, betweenness_dict, combined_scores, non_adjacent_nodes)
                    - ordered_nodes: List of non-adjacent nodes ordered by average of degree and betweenness centrality
                    - betweenness_dict: Dictionary with all non-adjacent nodes and their betweenness centrality
                    - combined_scores: Dictionary with combined scores for each non-adjacent node
                    - non_adjacent_nodes: Complete set of non-adjacent nodes
                """

                # # Convert root from tensor to integer if needed
                # if hasattr(root_node, 'item'):  # Check if it's a tensor
                #     root_node = root_node.item()
                # elif isinstance(root_node, (list, tuple)) and len(root_node) > 0:
                #     root_node = root_node[0]  # If it's a list/tuple with one element
                
                if root_node not in graph.nodes():
                    raise ValueError(f"Root node {root_node} not found in graph")
                
                # Get all non-adjacent nodes (nodes not connected to root and not the root itself)
                all_nodes = set(graph.nodes())
                adjacent_nodes = set(graph.neighbors(root_node))
                adjacent_nodes.add(root_node)  # Include root itself
                non_adjacent_nodes = all_nodes - adjacent_nodes
                
                # If no non-adjacent nodes exist
                if not non_adjacent_nodes:
                    return [], {}, {}, set()
                
                # Calculate betweenness centrality for all nodes
                betweenness = nx.betweenness_centrality(graph, normalized=True)
                
                # Get degree for all nodes
                degrees = dict(graph.degree())
                
                # Normalize degree values to 0-1 range for fair averaging with betweenness
                max_degree = max(degrees.values()) if degrees.values() else 1
                normalized_degrees = {node: deg / max_degree for node, deg in degrees.items()}
                
                # Calculate combined score (average of normalized degree and betweenness)
                combined_scores = {}
                for node in non_adjacent_nodes:
                    norm_degree = normalized_degrees.get(node, 0)
                    betw_centrality = betweenness.get(node, 0)
                    combined_scores[node] = (norm_degree + betw_centrality) / 2
                
                # Order non-adjacent nodes by combined score (descending order)
                ordered_nodes = sorted(non_adjacent_nodes, 
                                    key=lambda x: combined_scores[x], 
                                    reverse=True)
                
                # Create betweenness dictionary for non-adjacent nodes
                non_adjacent_betweenness = {node: betweenness.get(node, 0.0) for node in non_adjacent_nodes}
                
                return ordered_nodes, non_adjacent_betweenness, combined_scores, non_adjacent_nodes



        def calculate_ordered_all_nodes_betweenness(graph, root_node=None):
                """
                Calculate betweenness centrality for all nodes in the graph and order them
                based on the average of their normalized degree and betweenness centrality.
                
                Parameters:
                graph: NetworkX graph object containing the network
                root_node (int, optional): The root node (if specified, can be used for filtering or reference)
                
                Returns:
                tuple: (ordered_nodes, betweenness_dict, combined_scores, degrees_dict)
                    - ordered_nodes: List of ALL nodes ordered by average of degree and betweenness centrality
                    - betweenness_dict: Dictionary with ALL nodes and their betweenness centrality
                    - combined_scores: Dictionary with combined scores for ALL nodes
                    - degrees_dict: Dictionary with normalized degrees for ALL nodes
                """
                
                # Convert root from tensor to integer if needed
                if hasattr(root_node, 'item'):  # Check if it's a tensor
                    root_node = root_node.item()
                elif isinstance(root_node, (list, tuple)) and len(root_node) > 0:
                    root_node = root_node[0]  # If it's a list/tuple with one element
                    
                # Validate input
                if graph is None or len(graph.nodes()) == 0:
                    return [], {}, {}, {}
                
                if root_node is not None and root_node not in graph.nodes():
                    raise ValueError(f"Root node {root_node} not found in graph")
                
                # Calculate betweenness centrality for all nodes
                betweenness = nx.betweenness_centrality(graph, normalized=True)
                
                # Get degree for all nodes
                degrees = dict(graph.degree())
                
                # Normalize degree values to 0-1 range for fair averaging with betweenness
                max_degree = max(degrees.values()) if degrees.values() else 1
                normalized_degrees = {node: deg / max_degree for node, deg in degrees.items()}
                
                # Calculate combined score (average of normalized degree and betweenness) for ALL nodes
                combined_scores = {}
                for node in graph.nodes():
                    norm_degree = normalized_degrees.get(node, 0)
                    betw_centrality = betweenness.get(node, 0)
                    combined_scores[node] = (norm_degree + betw_centrality) / 2
                                   
                    ordered_nodes = sorted([node for node in graph.nodes() if node in combined_scores], 
                      key=lambda x: combined_scores[x], 
                                            reverse=True)
                   
                    
                    
                return ordered_nodes, betweenness, combined_scores, normalized_degrees
        

        
        
        def find_k_hop_neighbors(graph, start_node, k=1):
            """Finds all nodes within k steps from start_node"""
            all_neighbors = []
            current_level = {start_node}
            all_neighbors = {start_node}
                
            for _ in range(k):
                    next_level = set()
                    for node in current_level:
                        for neighbor in graph.neighbors(node):
                            if neighbor not in all_neighbors:
                                    next_level.add(neighbor)
                                    all_neighbors.add(neighbor)
                        current_level = next_level
                    
            return all_neighbors 
        


        def normalize_scores(scores: dict) -> dict:
            if not scores:
                return {}
            min_val = min(scores.values())
            max_val = max(scores.values())
            if max_val == min_val:
                return {node: 0.5 for node in scores}
            return {node: (score - min_val) / (max_val - min_val) for node, score in scores.items()}




        def get_ordered_nodes(nodes: List[int]):
            
            if len(nodes) <= 1:
                    return nodes, {}
            
            subgraph = self.graph.subgraph(nodes)
            degree_centrality = nx.degree_centrality(subgraph)
            betweenness_centrality = nx.betweenness_centrality(subgraph)
            
            normalized_degree = normalize_scores(degree_centrality)
            normalized_betweenness = normalize_scores(betweenness_centrality)
            
            combined_scores = {}
            for node in nodes:
                deg_score = normalized_degree.get(node, 0)
                bet_score = normalized_betweenness.get(node, 0)
                combined_scores[node] = (deg_score + bet_score) / 2
            
            # ordered_nodes = sorted(nodes, key=lambda x: combined_scores.get(x, 0), reverse=False)
            
            ordered_nodes = sorted([node for node in subgraph.nodes() if node in combined_scores], 
                      key=lambda x: combined_scores[x], reverse=False)
            
            return ordered_nodes, combined_scores
        

       

        def append_nodes_with_yield(root, node_set, end_count):
            """
            Yields the sequence after each iteration.
            """

            # Convert root from tensor to integer if needed
            if hasattr(root, 'item'):  # Check if it's a tensor
                root = root.item()
            elif isinstance(root, (list, tuple)) and len(root) > 0:
                root = root[0]  # If it's a list/tuple with one element

            available_nodes = list(node_set)
            current_sequence = [root]
            
            yield current_sequence.copy(), available_nodes.copy(), 0
            
            iteration = 1
            while len(current_sequence) < end_count +1 and available_nodes:
                selected_node = random.choice(available_nodes)
                available_nodes.remove(selected_node)
                current_sequence.append(selected_node)
                
                yield current_sequence.copy(), available_nodes.copy(), iteration
                iteration += 1

    
        # ***************************************************



        cur_graph_coalition = tree_node.coalition
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
        # L1 block
        
            if config.DargumentX == 'MODIFIED' : 
            #  Modified code version
            # -----------------------------
                node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
                # sort tuple according to an order i.e. low-to-high or vice versa
                # this is useful since higher degree nodes depict their importance in the subgraph
                # node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=self.high2low)
                # this is useful since higher degree nodes depict their importance in the subgraph, Higher to lower degree
            
                node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=True)
                # all_nodes holds only the IDs of nodes that have been ordered
                all_nodes = [x[0] for x in node_degree_list]
                # Nodes_WithRootNode = [x[0] for x in node_degree_list]
               

             
                # # Convert root from tensor to integer if needed
                # if hasattr(self.node_idx, 'item'):  # Check if it's a tensor
                #     self.node_idx = self.node_idx.item()
                # elif isinstance(self.node_idx, (list, tuple)) and len(self.node_idx) > 0:
                #     self.node_idx = self.node_idx[0]  # If it's a list/tuple with one element

                # self.node_idx typically represents a recently added node that should not be re-expalined
                # if self.node_idx:
                #     # this line excludes it newly added node
                #     expand_nodes = [node for node in all_nodes if node != self.node_idx]
                           
                # else:
                #     expand_nodes = all_nodes

                
                if len(all_nodes) < self.expand_atoms:   # added line
                    expand_nodes = all_nodes      # added line
                else:
                    expand_nodes = all_nodes[:self.expand_atoms]  # added line
                
               
                # # self.expand_atoms is the max set limit value which indicates the max number of children of a node. Truncation affects lower degrees 
                # if len(expand_nodes) > self.expand_atoms:
                #     expand_nodes = expand_nodes[:self.expand_atoms]
                
               
                
                for each_node in expand_nodes:

                    # Create a coalition excluding 'each_node'
                    subgraph_coalition = [node for node in all_nodes if node != each_node]

                       
                    # Generate connected subgraphs from the coalition
                    subgraphs = [self.graph.subgraph(c)
                                        for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]
                    
                    # If a 'node_idx' exists, find the subgraph containing it
                     
                  
                    main_sub = subgraphs[0]
                   
                    for sub in subgraphs:
                        if sub.number_of_nodes() > main_sub.number_of_nodes():
                                main_sub = sub
                                    




                    # Begin segregation/ differentiation from here, Graph-level and Node-level
                    if main_sub.number_of_nodes() > 0 :
                        # Graph-level

                        # ordered_nodes, betweenness, combined_scores, normalized_degrees = calculate_ordered_all_nodes_betweenness(main_sub)
                        # first_list, second_list = split_node_list_ordered(ordered_nodes,self.min_atoms -1 )
                        # nodes_to_process = list(second_list)[:config.NumOfCombinationSets]
                        # # nodes_to_process = list(second_list)[:len(second_list)]
                        
                                            
                        
                        # for node in nodes_to_process:
                        #     subgraph_coalitionz = list(first_list) + [node]

                        #     subgraphsx = [self.graph.subgraph(c)
                        #                                     for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                    
                        #     # If a 'new_node_idx' exists, find the subgraph containing it
                        #     if self.new_node_idx:
                        #                 for subx in subgraphsx:
                        #                     if self.new_node_idx in list(subx.nodes()):
                        #                         main_subx = subx
                        #     else:
                        #                 main_subx = subgraphsx[0]

                        #                 for subx in subgraphsx:
                        #                     if sub.number_of_nodes() > main_subx.number_of_nodes():
                        #                         main_subx = subx
                                                        

                        #     new_graph_coalitionx = sorted(list(main_subx.nodes()))
                        #                         # check the state map and merge the same sub-graph
                                    
                        #     find_same = False
                        #                         # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                        #                         # self.state_map is typically a dictionary mapping node IDs to their historical data.
                        #                         # this checks all MCTS nodes created so far fr equality
                        #     for old_graph_node in self.state_map.values():
                        #             if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                        #                     new_node = old_graph_node
                        #                     find_same = True

                        #                         # Only execute if no matching coalition was found
                        #     if not find_same:
                        #                             # Create new node
                        #             new_node = self.MCTSNodeClass(new_graph_coalitionx)
                        #                             # Store in state map (dictionary)
                        #             self.state_map[str(new_graph_coalitionx)] = new_node

                        #             find_same_child = False
                        #                         # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                        #     for cur_child in tree_node.children:
                        #                 if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                        #                         find_same_child = True

                        #     if not find_same_child:
                        #                 tree_node.children.append(new_node)
                        #                 ycounter+=1
                            
                        # # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                        # # The tree node's children contribution (P) is then computed using Shapley value function
                        # scores = compute_scores(self.score_func, tree_node.children)
                        # for child, score in zip(tree_node.children, scores):
                        #             child.P = score
                        #             ycounter =0 # reset child count
                        
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                        # ordered_nodes, betweenness, combined_scores, normalized_degrees = calculate_ordered_all_nodes_betweenness(main_sub)
                        # first_list, second_list = split_node_list_ordered(ordered_nodes,(len(ordered_nodes)-1)  )
                        # nodes_to_processu = list(first_list)[:(len(ordered_nodes)-1)]


                        # __________________++++++++++++++++++++++++++++++++________________________

                        # SubNodes_degree_list = list(self.graph.subgraph(main_sub.nodes()).degree)
                        # SubNodes_degree_list = sorted(SubNodes_degree_list, key=lambda x: x[1], reverse=True)
                        # SubNodes_degree_list_Orderlist = [x[0] for x in SubNodes_degree_list]
                        
                        # first_list, second_list = split_node_list_ordered(SubNodes_degree_list_Orderlist,(len(SubNodes_degree_list_Orderlist)-1)  )
                        # nodes_to_processu = list(first_list)[:(len(SubNodes_degree_list_Orderlist)-1)]



                        # __________________++++++++++++++++++++++++++++++++________________________

                    

                        
                        if config.SubgraphAlgoSel == 1 :
                        # Pick all subgraphs equal to or larger than the minimum coalition

                        # new_ordered, new_scores = get_ordered_nodes(main_sub.nodes())
                            new_ordered = list(main_sub.nodes())
                            nodes_to_processm = list(new_ordered)[:len(new_ordered)]

                                                        
                                                            
                            current_nodes = nodes_to_processm.copy()
                            current_nodesL = list(current_nodes)
                                    
                                
                                
                            while len(current_nodesL) >= config.Gmin_atoms:
                                
                                                                # Generate connected subgraphs from the coalition
                                    subgraphsx = [self.graph.subgraph(c)
                                                    for c in nx.connected_components(self.graph.subgraph(current_nodesL))]
                                                                                
                                                                
                                
                                    main_subx = subgraphsx[0]

                                    for subx in subgraphsx:
                                                if subx.number_of_nodes() >= config.Gmin_atoms:
                                                    main_subx = subx
                                                    
                                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                            # check the state map and merge the same sub-graph
                                                    find_same = False
                                                            # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                            # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                            # this checks all MCTS nodes created so far fr equality
                                                    for old_graph_node in self.state_map.values():
                                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                                new_node = old_graph_node
                                                                find_same = True

                                                            # Only execute if no matching coalition was found
                                                    if not find_same:
                                                                # Create new node
                                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                                # Store in state map (dictionary)
                                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                                    find_same_child = False
                                                            # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                    for cur_child in tree_node.children:
                                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                                find_same_child = True

                                                    if not find_same_child:
                                                            tree_node.children.append(new_node)
                                                            # ycounter+=1


                                    
                                    new_ordered, new_scores = get_ordered_nodes(current_nodesL)
                                    # new_ordered = current_nodesL
                                    first_list, second_list = split_node_list_ordered(new_ordered,1)
                                    
                                    current_nodesL = list(second_list)[:len(second_list)]
                                    
            
                            if  tree_node.children:
                                scores = compute_scores(self.score_func, tree_node.children)
                                for child, score in zip(tree_node.children, scores):
                                        child.P = score
                                        # ycounter =0 # reset child count
                                    
                        
                        elif config.SubgraphAlgoSel == 2 :
                        
                        # Pick the largest subgraph only

                            # new_ordered, new_scores = get_ordered_nodes(main_sub.nodes())
                            new_ordered = list(main_sub.nodes())
                            nodes_to_processm = list(new_ordered)[:len(new_ordered)]

                                                        
                                                            
                            current_nodes = nodes_to_processm.copy()
                            current_nodesL = list(current_nodes)
                                    
                                
                                
                            while len(current_nodesL) >= config.Gmin_atoms:
                                
                                                                # Generate connected subgraphs from the coalition
                                    subgraphsx = [self.graph.subgraph(c)
                                                    for c in nx.connected_components(self.graph.subgraph(current_nodesL))]
                                                                                
                                    main_subx = subgraphsx[0]                            
                                
                                    for subx in subgraphsx:
                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                    main_subx = subx
                                                    

                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                            # check the state map and merge the same sub-graph
                                    find_same = False
                                            # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                            # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                            # this checks all MCTS nodes created so far fr equality
                                    for old_graph_node in self.state_map.values():
                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                new_node = old_graph_node
                                                find_same = True

                                            # Only execute if no matching coalition was found
                                    if not find_same:
                                                # Create new node
                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                # Store in state map (dictionary)
                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                    find_same_child = False
                                            # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                    for cur_child in tree_node.children:
                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                find_same_child = True

                                    if not find_same_child:
                                            tree_node.children.append(new_node)
                                            # ycounter+=1


                                    
                                    new_ordered, new_scores = get_ordered_nodes(current_nodesL)
                                    # new_ordered = current_nodesL
                                    first_list, second_list = split_node_list_ordered(new_ordered,1)
                                    
                                    current_nodesL = list(second_list)[:len(second_list)]
                                    
            
                            if  tree_node.children:
                                scores = compute_scores(self.score_func, tree_node.children)
                                for child, score in zip(tree_node.children, scores):
                                        child.P = score
                                        # ycounter =0 # reset child count
                                    
                                
                                
                            

                        
                        elif config.SubgraphAlgoSel == 3 :

                            

                            ordered_nodes, betweenness, combined_scores, normalized_degrees = calculate_ordered_all_nodes_betweenness(main_sub)
                            first_list, second_list = split_node_list_ordered(ordered_nodes,(main_sub.number_of_nodes()))
                            nodes_to_processu = list(first_list)[:(main_sub.number_of_nodes())]
                        
                                                                    
                                            
                            
                            for nodeu in nodes_to_processu:

                    
                                BaseNum = self.min_atoms - 2
                                # NumOfCombinationSets = 3   # Number of combinations for form      
                                adjacent_nodes, remaining_nodes = get_adjacent_nodes_and_remaining(nodeu, main_sub.edges(), main_sub.nodes())
                                                
                                # Connected_subgraph_degree_list = list(self.graph.subgraph(main_sub.nodes()).degree)
                                # Connected_subgraph_degree_list = sorted(Connected_subgraph_degree_list, key=lambda x: x[1], reverse=self.high2low)
                            
                                Adjacent_NodesToRoot_degree_list = list(self.graph.subgraph(adjacent_nodes).degree)
                                Adjacent_NodesToRoot_degree_list = sorted(Adjacent_NodesToRoot_degree_list, key=lambda x: x[1], reverse=True)
                                Adjacent_NodesToRoot_degree_Orderlist = [x[0] for x in Adjacent_NodesToRoot_degree_list]

                        
                                # Remaining_NodesToRoot_degree_list = list(self.graph.subgraph(remaining_nodes).degree)
                                # Remaining_NodesToRoot_degree_list = sorted(Remaining_NodesToRoot_degree_list, key=lambda x: x[1], reverse=True)
                                # Remaining_NodesToRoot_degree_Orderlist = [x[0] for x in Remaining_NodesToRoot_degree_list]

                                # print(f"Original Subgraph nodes with roont note: {Nodes_WithRootNode}")
                                # print(f"Subgraph coalition after removing expand node : {subgraph_coalition}")
                                # print(f"Subgraph connected nodes: {main_sub.nodes()}")
                                # print(f"Removed expand node is : {each_node} and Adjacent nodes: {sorted(adjacent_nodes)}")
                                # print(f"Remaining nodes: {sorted(remaining_nodes)}")
                                # print(f"Connected subgraph node - degree list: {Connected_subgraph_degree_list}")
                                # print(f"Adjacent Nodes To Root degree list: {Adjacent_NodesToRoot_degree_list}")
                                # print(f"Adjacent Nodes To Root degree Order list: {Adjacent_NodesToRoot_degree_Orderlist}") 
                                # print(f"Remaining Nodes To Root degree list: {Remaining_NodesToRoot_degree_list}")
                                # print(f"Remaining Nodes To Root degree Order list: {Remaining_NodesToRoot_degree_Orderlist}")


                            
                                # Create a sample graph using the subgraph of which the root node is part
                                Gr = nx.Graph()
                                Gr.add_edges_from(main_sub.edges())
                                # Get adjacent nodes to root (node ..)
                                # adjacent_nodesx = list(Gr.neighbors(nodeu))
                                # print("Original Adjacent nodes:", adjacent_nodesx)
                                    
                                # Find an ordered list of adjacent nodes to the root node based on degree and betweenness closeness
                                ordered_nodesA, adjacent_betweenness, combined_scores = calculate_ordered_adjacent_nodes_betweenness(
                                        Adjacent_NodesToRoot_degree_Orderlist, Gr, nodeu)
                                
                                # print(f"Ordered Adjacent nodes to root list: {ordered_nodesA}") 
                                
                                # Find an ordered list of remaining nodes relative to the root node based on degree and betweenness closeness
                                ordered_nodes, betweenness_dict, combined_scores, non_adjacent_nodes = calculate_ordered_non_adjacent_nodes_betweenness(Gr, nodeu)
                                # print(f"Ordered Non-Adjacent nodes to root list: {ordered_nodes}") 
                        
                                    
                                if len(ordered_nodesA) < BaseNum :
                                # L2 block
                                    # Part of the non-adjacent node set will be used to form the base node since the adjacent set is not enough
                                                
                                    first_list, second_list = split_node_list_ordered(ordered_nodes,(BaseNum - len(ordered_nodesA)))
                                    BaseNodes = list(ordered_nodesA) + list(first_list)
                                    # second list becomes the remaining part
                                
                                    second_listcont = second_list
                                
                                    if len(second_listcont) <= config.NumOfCombinationSets:
                                    # L3 block
                                        first_list, second_list = split_node_list_ordered(second_listcont, len(second_listcont))
                                        nodes_to_process = list(first_list)[:len(second_listcont)]
                                        for node in nodes_to_process:
                                                    subgraph_coalitionz = [nodeu] + list(BaseNodes) + [node]
                                                    
                                                    # all_neighbors  = find_k_hop_neighbors(main_sub, nodeu, k=2)
                                                    # subgraph_coalitionz = list(set(subgraph_coalitionz).union(all_neighbors))

                                                    remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                    subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                                    # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                    # Generate connected subgraphs from the coalition
                                                    subgraphsx = [self.graph.subgraph(c)
                                                                    for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                    
                                                    main_subx = subgraphsx[0]                                        
                                                    # If a 'node_idx' exists, find the subgraph containing it
                                                
                                                    for subx in subgraphsx:
                                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                    main_subx = subx
                                                                

                                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                        # check the state map and merge the same sub-graph
                                                    find_same = False
                                                        # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                        # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                        # this checks all MCTS nodes created so far fr equality
                                                    for old_graph_node in self.state_map.values():
                                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                                new_node = old_graph_node
                                                                find_same = True

                                                        # Only execute if no matching coalition was found
                                                    if not find_same:
                                                            # Create new node
                                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                            # Store in state map (dictionary)
                                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                                    find_same_child = False
                                                        # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                    for cur_child in tree_node.children:
                                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                                find_same_child = True

                                                    if not find_same_child:
                                                            tree_node.children.append(new_node)
                                                            # ycounter+=1
                                        
                                        # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                        # The tree node's children contribution (P) is then computed using Shapley value function
                                        scores = compute_scores(self.score_func, tree_node.children)
                                        for child, score in zip(tree_node.children, scores):
                                                        child.P = score
                                                        # ycounter =0 # reset child count

                                    elif len(second_listcont) > config.NumOfCombinationSets:
                                    # L3 block
                                        first_list, second_list = split_node_list_ordered(second_listcont, config.NumOfCombinationSets)
                                        nodes_to_process = list(first_list)[:config.NumOfCombinationSets]
                                        for node in nodes_to_process:
                                                    subgraph_coalitionz = [nodeu] + list(BaseNodes) + [node]
                                                    
                                                    # all_neighbors  = find_k_hop_neighbors(main_sub, nodeu, k=2)
                                                    # subgraph_coalitionz = list(set(subgraph_coalitionz).union(all_neighbors))
                                                    remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                    subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                                    
                                                    # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                    # Generate connected subgraphs from the coalition
                                                    subgraphsx = [self.graph.subgraph(c)
                                                                    for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                            
                                                    main_subx = subgraphsx[0]
                                                    # If a 'node_idx' exists, find the subgraph containing it
                                                   
                                                    for subx in subgraphsx:
                                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                    main_subx = subx
                                                                

                                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                        # check the state map and merge the same sub-graph
                                                    find_same = False
                                                        # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                        # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                        # this checks all MCTS nodes created so far fr equality
                                                    for old_graph_node in self.state_map.values():
                                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                                new_node = old_graph_node
                                                                find_same = True

                                                        # Only execute if no matching coalition was found
                                                    if not find_same:
                                                            # Create new node
                                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                            # Store in state map (dictionary)
                                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                                    find_same_child = False
                                                        # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                    for cur_child in tree_node.children:
                                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                                find_same_child = True

                                                    if not find_same_child:
                                                            tree_node.children.append(new_node)
                                                            # ycounter+=1
                                        
                                        # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                        # The tree node's children contribution (P) is then computed using Shapley value function
                                        scores = compute_scores(self.score_func, tree_node.children)
                                        for child, score in zip(tree_node.children, scores):
                                                        child.P = score
                                                        # ycounter =0 # reset child count

                                elif len(ordered_nodesA) == BaseNum :
                                # L2 Block
                                    # Since the adjacent node set will just be enough to serve as the base nodes
                                    # then use the non-adjacent nodes for remaining part of the coalition
                                    # This time you are splitting the non-adjacent nodes
                                    BaseNodes = ordered_nodesA
                                    if len(ordered_nodes) <= config.NumOfCombinationSets :
                                    # L3 block
                                        NonAdjacentNodesx = ordered_nodes
                                        first_list, second_list = split_node_list_ordered(NonAdjacentNodesx, len(NonAdjacentNodesx))
                                        nodes_to_process = list(first_list)[:len(NonAdjacentNodesx)]
                                        for node in nodes_to_process:
                                                    subgraph_coalitionz = [nodeu] + list(BaseNodes) + [node]

                                                    remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                    subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                                    
                                                
                                                    # all_neighbors  = find_k_hop_neighbors(main_sub, nodeu, k=2)
                                                    # subgraph_coalitionz = list(set(subgraph_coalitionz).union(all_neighbors))
                                                    
                                                    # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                    # Generate connected subgraphs from the coalition
                                                    subgraphsx = [self.graph.subgraph(c)
                                                                    for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                    
                                                    main_subx = subgraphsx[0] 

                                                    # If a 'node_idx' exists, find the subgraph containing it
                                                    
                                                    for subx in subgraphsx:
                                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                   main_subx = subx
                                                                

                                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                        # check the state map and merge the same sub-graph
                                                    find_same = False
                                                        # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                        # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                        # this checks all MCTS nodes created so far fr equality
                                                    for old_graph_node in self.state_map.values():
                                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                                new_node = old_graph_node
                                                                find_same = True

                                                        # Only execute if no matching coalition was found
                                                    if not find_same:
                                                            # Create new node
                                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                            # Store in state map (dictionary)
                                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                                    find_same_child = False
                                                        # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                    for cur_child in tree_node.children:
                                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                                find_same_child = True

                                                    if not find_same_child:
                                                            tree_node.children.append(new_node)
                                                            # ycounter+=1
                                        
                                        # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                        # The tree node's children contribution (P) is then computed using Shapley value function
                                        scores = compute_scores(self.score_func, tree_node.children)
                                        for child, score in zip(tree_node.children, scores):
                                                        child.P = score
                                                        # ycounter =0 # reset child count

                                    elif len(ordered_nodes) > config.NumOfCombinationSets :
                                    # L3 block
                                        NonAdjacentNodesx = ordered_nodes          
                                        first_list, second_list = split_node_list_ordered(NonAdjacentNodesx, config.NumOfCombinationSets)
                                        nodes_to_process = list(first_list)[:config.NumOfCombinationSets]
                                        for node in nodes_to_process:
                                                    subgraph_coalitionz = [nodeu] + list(BaseNodes) + [node]

                                                    remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                    subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                                    
                                                
                                                    # all_neighbors  = find_k_hop_neighbors(main_sub, nodeu, k=2)
                                                    # subgraph_coalitionz = list(set(subgraph_coalitionz).union(all_neighbors))
                                                    
                                                    # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                    # Generate connected subgraphs from the coalition
                                                    subgraphsx = [self.graph.subgraph(c)
                                                                    for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                            
                                                    main_subx = subgraphsx[0]
                                                    # If a 'node_idx' exists, find the subgraph containing it
                                                

                                                    for subx in subgraphsx:
                                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                   main_subx = subx
                                                                

                                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                        # check the state map and merge the same sub-graph
                                                    find_same = False
                                                        # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                        # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                        # this checks all MCTS nodes created so far fr equality
                                                    for old_graph_node in self.state_map.values():
                                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                                new_node = old_graph_node
                                                                find_same = True

                                                        # Only execute if no matching coalition was found
                                                    if not find_same:
                                                            # Create new node
                                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                            # Store in state map (dictionary)
                                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                                    find_same_child = False
                                                        # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                    for cur_child in tree_node.children:
                                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                                find_same_child = True

                                                    if not find_same_child:
                                                            tree_node.children.append(new_node)
                                                            # ycounter+=1
                                                    # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                                    # The tree node's children contribution (P) is then computed using Shapley value function
                                                    scores = compute_scores(self.score_func, tree_node.children)
                                                    for child, score in zip(tree_node.children, scores):
                                                        child.P = score
                                                        # ycounter =0 # reset child count
                                    
                                    
                                elif len(ordered_nodesA) > BaseNum :
                                # L2 block
                                        # case of nodes in the adjacent set being more than number required for base nodes
                                    if (len(ordered_nodesA) - BaseNum) < config.NumOfCombinationSets :
                                    # L3 block
                                        # If the remaining adjacent node set after forming the base is less then the combinatorial required number
                                        # then use the remaining non-adjacent node set after exhausting the remaining adjacent set
                                        # first_list is the base node set
                                        # second_list is the remaining set of adjacent nodes
                                        AdjacentNodesx = ordered_nodesA
                                        first_list, second_list = split_node_list_ordered(AdjacentNodesx, BaseNum)
                                        nodes_to_process = list(second_list)[:config.NumOfCombinationSets] # Note this set will not be enough
                                        BaseNodes = first_list
                                        for node in nodes_to_process:
                                            subgraph_coalitionz = [nodeu] + list(BaseNodes) + [node]
                                        
                                            remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                            subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                                    
                                            # all_neighbors  = find_k_hop_neighbors(main_sub, nodeu, k=2)
                                            # subgraph_coalitionz = list(set(subgraph_coalitionz).union(all_neighbors))
                                                    
                                        
                                            #   print(f"Adjacent nodes coalition: {subgraph_coalitionz}") 
                                            # Generate connected subgraphs from the coalition
                                            subgraphsx = [self.graph.subgraph(c)
                                                            for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                    
                                            main_subx = subgraphsx[0]
                                            # If a 'node_idx' exists, find the subgraph containing it
                                            
                                            for subx in subgraphsx:
                                                        if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                           main_subx = subx
                                                        

                                            new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                # check the state map and merge the same sub-graph
                                            find_same = False
                                                # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                # this checks all MCTS nodes created so far fr equality
                                            for old_graph_node in self.state_map.values():
                                                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                        new_node = old_graph_node
                                                        find_same = True

                                                # Only execute if no matching coalition was found
                                            if not find_same:
                                                    # Create new node
                                                    new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                    # Store in state map (dictionary)
                                                    self.state_map[str(new_graph_coalitionx)] = new_node

                                            find_same_child = False
                                                # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                            for cur_child in tree_node.children:
                                                    if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                        find_same_child = True

                                            if not find_same_child:
                                                    tree_node.children.append(new_node)
                                                    # ycounter+=1
                                        
                                        # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                        # The tree node's children contribution (P) is then computed using Shapley value function
                                        scores = compute_scores(self.score_func, tree_node.children)
                                        for child, score in zip(tree_node.children, scores):
                                                child.P = score
                                                # ycounter =0 # reset child count

                                    
                                        # This time you are splitting the non-adjacent nodes
                                        if len(ordered_nodes) <= (config.NumOfCombinationSets-(len(ordered_nodesA) - BaseNum)):
                                                    first_list, second_list = split_node_list_ordered(ordered_nodes, len(ordered_nodes))
                                                    nodes_to_process = list(first_list)[:len(ordered_nodes)] 
                                                    for node in nodes_to_process:
                                                        subgraph_coalitionz = [nodeu] + list(BaseNodes) + [node]
                                                    
                                                        
                                                        remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                        subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                                    
                                                        # all_neighbors  = find_k_hop_neighbors(main_sub, nodeu, k=2)
                                                        # subgraph_coalitionz = list(set(subgraph_coalitionz).union(all_neighbors))
                                                    
                                                    
                                                        # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                        # Generate connected subgraphs from the coalition
                                                        subgraphsx = [self.graph.subgraph(c)
                                                                    for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                            
                                                        main_subx = subgraphsx[0]
                                                        # If a 'node_idx' exists, find the subgraph containing it
                                                       
                                                        for subx in subgraphsx:
                                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                   main_subx = subx
                                                                

                                                        new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                        # check the state map and merge the same sub-graph
                                                        find_same = False
                                                        # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                        # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                        # this checks all MCTS nodes created so far fr equality
                                                        for old_graph_node in self.state_map.values():
                                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                                new_node = old_graph_node
                                                                find_same = True

                                                        # Only execute if no matching coalition was found
                                                        if not find_same:
                                                            # Create new node
                                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                            # Store in state map (dictionary)
                                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                                        find_same_child = False
                                                        # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                        for cur_child in tree_node.children:
                                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                                find_same_child = True

                                                        if not find_same_child:
                                                            tree_node.children.append(new_node)
                                                            # ycounter+=1
                                                    # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                                    # The tree node's children contribution (P) is then computed using Shapley value function
                                                    scores = compute_scores(self.score_func, tree_node.children)
                                                    for child, score in zip(tree_node.children, scores):
                                                            child.P = score
                                                            # ycounter =0 # reset child count

                                        elif len(ordered_nodes) > (config.NumOfCombinationSets-(len(ordered_nodesA) - BaseNum)):
                                                    AdjacentNodesx = ordered_nodesA
                                                    NonAdjacentNodesx = ordered_nodes
                                                    first_list, second_list = split_node_list_ordered(NonAdjacentNodesx, config.NumOfCombinationSets-(len(AdjacentNodesx) - BaseNum))
                                                    nodes_to_process = list(first_list)[:config.NumOfCombinationSets -(len(AdjacentNodesx) - BaseNum)] # Note this set will not be enough
                                                    for node in nodes_to_process:
                                                        subgraph_coalitionz = [nodeu] + list(BaseNodes) + [node]
                                                    
                                                       
                                                        remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                        subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                                    
                                                        # all_neighbors  = find_k_hop_neighbors(main_sub, nodeu, k=2)
                                                        # subgraph_coalitionz = list(set(subgraph_coalitionz).union(all_neighbors))
                                                    
                                                        # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                        # Generate connected subgraphs from the coalition
                                                        subgraphsx = [self.graph.subgraph(c)
                                                                        for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                                
                                                       
                                                        main_subx = subgraphsx[0]
                                                        
                                                        # If a 'node_idx' exists, find the subgraph containing it
                                                       
                                                        for subx in subgraphsx:
                                                                    if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                       main_subx = subx
                                                                    

                                                        new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                            # check the state map and merge the same sub-graph
                                                        find_same = False
                                                            # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                            # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                            # this checks all MCTS nodes created so far fr equality
                                                        for old_graph_node in self.state_map.values():
                                                                if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                                    new_node = old_graph_node
                                                                    find_same = True

                                                            # Only execute if no matching coalition was found
                                                        if not find_same:
                                                                # Create new node
                                                                new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                                # Store in state map (dictionary)
                                                                self.state_map[str(new_graph_coalitionx)] = new_node

                                                        find_same_child = False
                                                            # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                        for cur_child in tree_node.children:
                                                                if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                                    find_same_child = True

                                                        if not find_same_child:
                                                                tree_node.children.append(new_node)
                                                                # ycounter+=1
                                                    
                                                    
                                                    # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                                    # The tree node's children contribution (P) is then computed using Shapley value function
                                                    scores = compute_scores(self.score_func, tree_node.children)
                                                    for child, score in zip(tree_node.children, scores):
                                                            child.P = score
                                                            # ycounter =0 # reset child count
                                    
                                                                        
                                    elif len(ordered_nodesA) - BaseNum  >= config.NumOfCombinationSets :
                                    # L3 block
                                        # first_list is the base node set
                                        # second_list is the remaining set of adjacent nodes
                                        AdjacentNodesx = ordered_nodesA
                                        first_list, second_list = split_node_list_ordered(AdjacentNodesx, BaseNum)
                                        nodes_to_process = list(second_list)[:config.NumOfCombinationSets]
                                        for node in nodes_to_process:
                                            subgraph_coalitionz = [nodeu] + list(first_list) + [node]
                                        
                                            
                                            remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                            subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                                    
                                            # all_neighbors  = find_k_hop_neighbors(main_sub, nodeu, k=2)
                                            # subgraph_coalitionz = list(set(subgraph_coalitionz).union(all_neighbors))
                                                
                                        
                                        
                                            # Generate connected subgraphs from the coalition
                                            subgraphsx = [self.graph.subgraph(c)
                                                            for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                    
                                            main_subx = subgraphsx[0]
                                            # If a 'node_idx' exists, find the subgraph containing it
                                            
                                            for subx in subgraphsx:
                                                        if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                           main_subx = subx
                                                        

                                            new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                # check the state map and merge the same sub-graph
                                            find_same = False
                                                # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                # this checks all MCTS nodes created so far fr equality
                                            for old_graph_node in self.state_map.values():
                                                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                        new_node = old_graph_node
                                                        find_same = True

                                                # Only execute if no matching coalition was found
                                            if not find_same:
                                                    # Create new node
                                                    new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                    # Store in state map (dictionary)
                                                    self.state_map[str(new_graph_coalitionx)] = new_node

                                            find_same_child = False
                                                # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                            for cur_child in tree_node.children:
                                                    if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                        find_same_child = True

                                            if not find_same_child:
                                                    tree_node.children.append(new_node)
                                                    # ycounter+=1

                            
                            # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                            # The tree node's children contribution (P) is then computed using Shapley value function
                            scores = compute_scores(self.score_func, tree_node.children)
                            for child, score in zip(tree_node.children, scores):
                                        child.P = score
                                        # ycounter =0 # reset child count    
                    
                    elif self.node_idx != None and main_sub !=None:
                        if config.SubgraphAlgoSel == 1 :
                            # Pruning * and Ego - Network strictly from the adjcent nodes
                            # Node-level
                            BaseNum = self.min_atoms - 1 # because 1 slot is reserved for the root / ego node
                            # NumOfCombinationSets = 3   # Number of combinations for form      
                            adjacent_nodes, remaining_nodes = get_adjacent_nodes_and_remaining(self.node_idx, main_sub.edges(), main_sub.nodes())
                                                
                            # Connected_subgraph_degree_list = list(self.graph.subgraph(main_sub.nodes()).degree)
                            # Connected_subgraph_degree_list = sorted(Connected_subgraph_degree_list, key=lambda x: x[1], reverse=self.high2low)
                            Adjacent_NodesToRoot_degree_list = list(self.graph.subgraph(adjacent_nodes).degree)
                            Adjacent_NodesToRoot_degree_list = sorted(Adjacent_NodesToRoot_degree_list, key=lambda x: x[1], reverse=True)
                            Adjacent_NodesToRoot_degree_Orderlist = [x[0] for x in Adjacent_NodesToRoot_degree_list]
                            # Remaining_NodesToRoot_degree_list = list(self.graph.subgraph(remaining_nodes).degree)
                            # Remaining_NodesToRoot_degree_list = sorted(Remaining_NodesToRoot_degree_list, key=lambda x: x[1], reverse=True)
                            # Remaining_NodesToRoot_degree_Orderlist = [x[0] for x in Remaining_NodesToRoot_degree_list]

                            # print(f"Original Subgraph nodes with roont note: {Nodes_WithRootNode}")
                            # print(f"Subgraph coalition after removing expand node : {subgraph_coalition}")
                            # print(f"Subgraph connected nodes: {main_sub.nodes()}")
                            # print(f"Removed expand node is : {each_node} and Adjacent nodes: {sorted(adjacent_nodes)}")
                            # print(f"Remaining nodes: {sorted(remaining_nodes)}")
                            # print(f"Connected subgraph node - degree list: {Connected_subgraph_degree_list}")
                            # print(f"Adjacent Nodes To Root degree list: {Adjacent_NodesToRoot_degree_list}")
                            # print(f"Adjacent Nodes To Root degree Order list: {Adjacent_NodesToRoot_degree_Orderlist}") 
                            # print(f"Remaining Nodes To Root degree list: {Remaining_NodesToRoot_degree_list}")
                            # print(f"Remaining Nodes To Root degree Order list: {Remaining_NodesToRoot_degree_Orderlist}")


                        
                            # Create a sample graph using the subgraph of which the root node is part
                            Gr = nx.Graph()
                            Gr.add_edges_from(main_sub.edges())
                            # Get adjacent nodes to root (node ..)
                            # adjacent_nodesx = list(Gr.neighbors(self.new_node_idx))
                            # print("Original Adjacent nodes:", adjacent_nodesx)
                                
                            # Find an ordered list of adjacent nodes to the root node based on degree and betweenness closeness
                            ordered_nodesA, adjacent_betweenness, combined_scores = calculate_ordered_adjacent_nodes_betweenness(
                                    Adjacent_NodesToRoot_degree_Orderlist, Gr, self.node_idx)
                            
                            # print(f"Ordered Adjacent nodes to root list: {ordered_nodesA}") 
                            
                            # Find an ordered list of remaining nodes relative to the root node based on degree and betweenness closeness
                            # ordered_nodes, betweenness_dict, combined_scores, non_adjacent_nodes = calculate_ordered_non_adjacent_nodes_betweenness(Gr, self.new_node_idx)
                            # print(f"Ordered Non-Adjacent nodes to root list: {ordered_nodes}") 

                            AdjacentNodesx = ordered_nodesA
                            first_list, second_list = split_node_list_ordered(AdjacentNodesx, BaseNum)
                            BaseNodes = first_list
                        
                            subgraph_coalitionz = [self.node_idx] + list(BaseNodes) 
                            remnamt = [item for item in remaining_nodes if item not in subgraph_coalitionz]
                            subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                            #   print(f"Adjacent nodes coalition: {subgraph_coalitionz}") 
                            # Generate connected subgraphs from the coalition
                            subgraphsx = [self.graph.subgraph(c)
                                            for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                    
                            # If a 'new_node_idx' exists, find the subgraph containing it
                            main_subx = subgraphsx[0]
                            
                            if self.node_idx:
                                    for subx in subgraphsx:
                                        if self.node_idx in list(subx.nodes()):
                                            main_subx = subx
                            else:
                                    

                                    for subx in subgraphsx:
                                        if subx.number_of_nodes() > main_subx.number_of_nodes():
                                            main_subx = subx
                                        

                            new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                # check the state map and merge the same sub-graph
                            find_same = False
                                # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                # this checks all MCTS nodes created so far fr equality
                            for old_graph_node in self.state_map.values():
                                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                        new_node = old_graph_node
                                        find_same = True

                                # Only execute if no matching coalition was found
                            if not find_same:
                                    # Create new node
                                    new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                    # Store in state map (dictionary)
                                    self.state_map[str(new_graph_coalitionx)] = new_node

                            find_same_child = False
                                # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                            for cur_child in tree_node.children:
                                    if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                        find_same_child = True

                            if not find_same_child:
                                    tree_node.children.append(new_node)
                                    # ycounter+=1
                            
                            # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                            # The tree node's children contribution (P) is then computed using Shapley value function
                            scores = compute_scores(self.score_func, tree_node.children)
                            for child, score in zip(tree_node.children, scores):
                                    child.P = score
                                    # ycounter =0 # reset child count

                        elif config.SubgraphAlgoSel == 2 : 
                            # Pruning * , Ego - Network  ( using the first and second order)
                            # Node-level
                            BaseNum = self.min_atoms - 1 # because 1 slot is reserved for the root / ego node
                            adjacent_nodes, remaining_nodes = get_adjacent_nodes_and_remaining(self.node_idx, main_sub.edges(), main_sub.nodes())
                                            
                            Adjacent_NodesToRoot_degree_list = list(self.graph.subgraph(adjacent_nodes).degree)
                            Adjacent_NodesToRoot_degree_list = sorted(Adjacent_NodesToRoot_degree_list, key=lambda x: x[1], reverse=True)
                            Adjacent_NodesToRoot_degree_Orderlist = [x[0] for x in Adjacent_NodesToRoot_degree_list]
                            # ordered_nodesA = list(Adjacent_NodesToRoot_degree_Orderlist) # this is the agjacent set, randomly pick nodes from here without replacement

                            ordered_nodesA = list(Adjacent_NodesToRoot_degree_Orderlist) + list(remaining_nodes)
                        
                            if len(ordered_nodesA) < BaseNum :
                                BaseNum = len(ordered_nodesA)

                            # Manual control with next()
                            generator = append_nodes_with_yield(self.node_idx, ordered_nodesA, BaseNum)                         
                        

                            try:
                                # Get initial state
                                sequence, remaining, iteration_num = next(generator)
                                # print(f"Initial: {sequence}")
                                
                                # Process each iteration manually
                                while True:
                                    sequence, remaining, iteration_num = next(generator)
                                    # print(f"Iteration {iteration_num}: current_sequence = {sequence}")
                                    
                                    subgraph_coalitionz = sequence

                                    remnamt = [item for item in remaining_nodes if item not in subgraph_coalitionz]
                                    subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)
                                    # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                    # Generate connected subgraphs from the coalition
                                    subgraphsx = [self.graph.subgraph(c)
                                                    for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                            
                                    # If a 'node_idx' exists, find the subgraph containing it
                                    main_subx = subgraphsx[0]
                                    
                                    if self.node_idx:
                                            for subx in subgraphsx:
                                                if self.node_idx in list(subx.nodes()):
                                                    main_subx = subx
                                    else:
                                           
                                        for subx in subgraphsx:
                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                    main_subx = subx
                                                

                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                        # check the state map and merge the same sub-graph
                                    find_same = False
                                        # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                        # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                        # this checks all MCTS nodes created so far fr equality
                                    for old_graph_node in self.state_map.values():
                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                new_node = old_graph_node
                                                find_same = True

                                        # Only execute if no matching coalition was found
                                    if not find_same:
                                            # Create new node
                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                            # Store in state map (dictionary)
                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                    find_same_child = False
                                        # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                    for cur_child in tree_node.children:
                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                find_same_child = True

                                    if not find_same_child:
                                            tree_node.children.append(new_node)
                                            # ycounter+=1
                                    
                                    
                                    # You can add conditions to break early if needed
                                    if len(sequence) > BaseNum:  # Stop after BaseNum nodes for example
                                        print("Stopping early at iteration", iteration_num)
                                        break

                                    # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                    # The tree node's children contribution (P) is then computed using Shapley value function
                                    scores = compute_scores(self.score_func, tree_node.children)
                                    for child, score in zip(tree_node.children, scores):
                                                    child.P = score
                                                    # ycounter =0 # reset child count

                                        
                            except StopIteration:
                                print("All iterations completed")
                                

                        elif config.SubgraphAlgoSel == 3 : 
                            # Pruning , Ego - Network and Combinatorial   

                            # Node-level
                            BaseNum = self.min_atoms - 2
                            # NumOfCombinationSets = 3   # Number of combinations for form 
                            # print(f"Adjacent_NodesToRoot_degree_Orderlist list: {Adjacent_NodesToRoot_degree_Orderlist}")     
                            adjacent_nodes, remaining_nodes = get_adjacent_nodes_and_remaining(self.node_idx, main_sub.edges(), main_sub.nodes())
                                                
                           
                            # Connected_subgraph_degree_list = list(self.graph.subgraph(main_sub.nodes()).degree)
                            # Connected_subgraph_degree_list = sorted(Connected_subgraph_degree_list, key=lambda x: x[1], reverse=self.high2low)
                            Adjacent_NodesToRoot_degree_list = list(self.graph.subgraph(adjacent_nodes).degree)
                            Adjacent_NodesToRoot_degree_list = sorted(Adjacent_NodesToRoot_degree_list, key=lambda x: x[1], reverse=True)
                            Adjacent_NodesToRoot_degree_Orderlist = [x[0] for x in Adjacent_NodesToRoot_degree_list]
                            # Remaining_NodesToRoot_degree_list = list(self.graph.subgraph(remaining_nodes).degree)
                            # Remaining_NodesToRoot_degree_list = sorted(Remaining_NodesToRoot_degree_list, key=lambda x: x[1], reverse=True)
                            # Remaining_NodesToRoot_degree_Orderlist = [x[0] for x in Remaining_NodesToRoot_degree_list]

                            # print(f"Original Subgraph nodes with roont note: {Nodes_WithRootNode}")
                            # print(f"Subgraph coalition after removing expand node : {subgraph_coalition}")
                            # print(f"Subgraph connected nodes: {main_sub.nodes()}")
                            # print(f"Removed expand node is : {each_node} and Adjacent nodes: {sorted(adjacent_nodes)}")
                            # print(f"Remaining nodes: {sorted(remaining_nodes)}")
                            # print(f"Connected subgraph node - degree list: {Connected_subgraph_degree_list}")
                            # print(f"Adjacent Nodes To Root degree list: {Adjacent_NodesToRoot_degree_list}")
                            # print(f"Adjacent Nodes To Root degree Order list: {Adjacent_NodesToRoot_degree_Orderlist}") 
                            # print(f"Remaining Nodes To Root degree list: {Remaining_NodesToRoot_degree_list}")
                            # print(f"Remaining Nodes To Root degree Order list: {Remaining_NodesToRoot_degree_Orderlist}")

                      
                        
                            # Create a sample graph using the subgraph of which the root node is part
                            Gr = nx.Graph()
                            Gr.add_edges_from(main_sub.edges())
                            # Get adjacent nodes to root (node ..)
                            # adjacent_nodesx = list(Gr.neighbors(self.new_node_idx))
                            # print("Original Adjacent nodes:", adjacent_nodesx)

                            # print(Gr.nodes)
                            # print(Gr.edges)
                            # print(self.node_idx)   
                            # Find an ordered list of adjacent nodes to the root node based on degree and betweenness closeness
                            ordered_nodesA, adjacent_betweenness, combined_scores = calculate_ordered_adjacent_nodes_betweenness(
                                    Adjacent_NodesToRoot_degree_Orderlist, Gr, self.node_idx)
                            

                            
                            # Find an ordered list of remaining nodes relative to the root node based on degree and betweenness closeness
                            ordered_nodes, betweenness_dict, combined_scores, non_adjacent_nodes = calculate_ordered_non_adjacent_nodes_betweenness(Gr, self.node_idx)
                            # print(f"Ordered Non-Adjacent nodes to root list: {ordered_nodes}") 
                            
                           
                                
                            if len(ordered_nodesA) < BaseNum :
                            # L2 block
                                # Part of the non-adjacent node set will be used to form the base node since the adjacent set is not enough
                                            
                                first_list, second_list = split_node_list_ordered(ordered_nodes,(BaseNum - len(ordered_nodesA)))
                                BaseNodes = list(ordered_nodesA) + list(first_list)
                                # second list becomes the remaining part
                            
                                second_listcont = second_list
                            
                                if len(second_listcont) <= config.NumOfCombinationSets:
                                # L3 block
                                    first_list, second_list = split_node_list_ordered(second_listcont, len(second_listcont))
                                    nodes_to_process = list(first_list)[:len(second_listcont)]
                                    for node in nodes_to_process:
                                               
                                                subgraph_coalitionz = [self.node_idx] + list(BaseNodes) + [node]
                                               

                                                remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                                                 # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                # Generate connected subgraphs from the coalition
                                                subgraphsx = [self.graph.subgraph(c)
                                                                for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                        
                                               
                                                main_subx = subgraphsx[0]
                                                                                               
                                              
                                                # If a 'node_idx' exists, find the subgraph containing it
                                                if self.node_idx != None :
                                                        for subx in subgraphsx:
                                                            if self.node_idx in list(subx.nodes()):
                                                                main_subx = subx
                                                               
                                                else:
                                                        for subx in subgraphsx:
                                                            if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                main_subx = subx
                                                                
                                                            
                                               

                                                new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                    # check the state map and merge the same sub-graph
                                                find_same = False
                                                    # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                    # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                    # this checks all MCTS nodes created so far fr equality
                                                for old_graph_node in self.state_map.values():
                                                        if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                            new_node = old_graph_node
                                                            find_same = True

                                                    # Only execute if no matching coalition was found
                                                if not find_same:
                                                        # Create new node
                                                        new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                        # Store in state map (dictionary)
                                                        self.state_map[str(new_graph_coalitionx)] = new_node

                                                find_same_child = False
                                                    # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                for cur_child in tree_node.children:
                                                        if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                            find_same_child = True

                                                if not find_same_child:
                                                        tree_node.children.append(new_node)
                                                        # ycounter+=1
                                    
                                    # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                    # The tree node's children contribution (P) is then computed using Shapley value function
                                    scores = compute_scores(self.score_func, tree_node.children)
                                    for child, score in zip(tree_node.children, scores):
                                                    child.P = score
                                                    # ycounter =0 # reset child count

                                elif len(second_listcont) > config.NumOfCombinationSets:
                                # L3 block
                                    first_list, second_list = split_node_list_ordered(second_listcont, config.NumOfCombinationSets)
                                    nodes_to_process = list(first_list)[:config.NumOfCombinationSets]
                                    for node in nodes_to_process:
                                           
                                                subgraph_coalitionz = [self.node_idx] + list(BaseNodes) + [node]
                                                # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                # Generate connected subgraphs from the coalition

                                                remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                                                subgraphsx = [self.graph.subgraph(c)
                                                                for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                        
                                                main_subx = subgraphsx[0]
                                                                                               
                                               

                                                # If a 'node_idx' exists, find the subgraph containing it
                                                if self.node_idx != None :
                                                        for subx in subgraphsx:
                                                            if self.node_idx in list(subx.nodes()):
                                                                main_subx = subx
                                                               
                                                else:
                                                        for subx in subgraphsx:
                                                            if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                main_subx = subx
                                                                
                                                            

                                                new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                    # check the state map and merge the same sub-graph
                                                find_same = False
                                                    # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                    # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                    # this checks all MCTS nodes created so far fr equality
                                                for old_graph_node in self.state_map.values():
                                                        if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                            new_node = old_graph_node
                                                            find_same = True

                                                    # Only execute if no matching coalition was found
                                                if not find_same:
                                                        # Create new node
                                                        new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                        # Store in state map (dictionary)
                                                        self.state_map[str(new_graph_coalitionx)] = new_node

                                                find_same_child = False
                                                    # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                for cur_child in tree_node.children:
                                                        if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                            find_same_child = True

                                                if not find_same_child:
                                                        tree_node.children.append(new_node)
                                                        # ycounter+=1
                                    
                                    # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                    # The tree node's children contribution (P) is then computed using Shapley value function
                                    scores = compute_scores(self.score_func, tree_node.children)
                                    for child, score in zip(tree_node.children, scores):
                                                    child.P = score
                                                    # ycounter =0 # reset child count

                            elif len(ordered_nodesA) == BaseNum :
                            # L2 Block
                                # Since the adjacent node set will just be enough to serve as the base nodes
                                # then use the non-adjacent nodes for remaining part of the coalition
                                # This time you are splitting the non-adjacent nodes
                                BaseNodes = ordered_nodesA
                                if len(ordered_nodes) <= config.NumOfCombinationSets :
                                # L3 block
                                    NonAdjacentNodesx = ordered_nodes
                                    first_list, second_list = split_node_list_ordered(NonAdjacentNodesx, len(NonAdjacentNodesx))
                                    nodes_to_process = list(first_list)[:len(NonAdjacentNodesx)]
                                    for node in nodes_to_process:
                                                
                                                subgraph_coalitionz = [self.node_idx] + list(BaseNodes) + [node]
                                                # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                # Generate connected subgraphs from the coalition

                                                remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                                                subgraphsx = [self.graph.subgraph(c)
                                                                for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                        
                                                main_subx = subgraphsx[0]
                                                                                               
                                              
                                                # If a 'node_idx' exists, find the subgraph containing it
                                                if self.node_idx != None :
                                                        for subx in subgraphsx:
                                                            if self.node_idx in list(subx.nodes()):
                                                                main_subx = subx
                                                               
                                                else:
                                                        for subx in subgraphsx:
                                                            if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                main_subx = subx
                                                                
                                                            

                                                new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                    # check the state map and merge the same sub-graph
                                                find_same = False
                                                    # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                    # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                    # this checks all MCTS nodes created so far fr equality
                                                for old_graph_node in self.state_map.values():
                                                        if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                            new_node = old_graph_node
                                                            find_same = True

                                                    # Only execute if no matching coalition was found
                                                if not find_same:
                                                        # Create new node
                                                        new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                        # Store in state map (dictionary)
                                                        self.state_map[str(new_graph_coalitionx)] = new_node

                                                find_same_child = False
                                                    # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                for cur_child in tree_node.children:
                                                        if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                            find_same_child = True

                                                if not find_same_child:
                                                        tree_node.children.append(new_node)
                                                        # ycounter+=1
                                    
                                    # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                    # The tree node's children contribution (P) is then computed using Shapley value function
                                    scores = compute_scores(self.score_func, tree_node.children)
                                    for child, score in zip(tree_node.children, scores):
                                                    child.P = score
                                                    # ycounter =0 # reset child count

                                elif len(ordered_nodes) > config.NumOfCombinationSets :
                                # L3 block
                                    NonAdjacentNodesx = ordered_nodes          
                                    first_list, second_list = split_node_list_ordered(NonAdjacentNodesx, config.NumOfCombinationSets)
                                    nodes_to_process = list(first_list)[:config.NumOfCombinationSets]
                                    for node in nodes_to_process:
                                                
                                                subgraph_coalitionz = [self.node_idx] + list(BaseNodes) + [node]
                                                # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                # Generate connected subgraphs from the coalition

                                                remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                                                subgraphsx = [self.graph.subgraph(c)
                                                                for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                        
                                                main_subx = subgraphsx[0]
                                                                                               
                                               
                                                # If a 'node_idx' exists, find the subgraph containing it
                                                if self.node_idx != None :
                                                        for subx in subgraphsx:
                                                            if self.node_idx in list(subx.nodes()):
                                                                main_subx = subx
                                                               
                                                else:
                                                        for subx in subgraphsx:
                                                            if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                main_subx = subx
                                                                
                                                            

                                                new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                    # check the state map and merge the same sub-graph
                                                find_same = False
                                                    # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                    # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                    # this checks all MCTS nodes created so far fr equality
                                                for old_graph_node in self.state_map.values():
                                                        if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                            new_node = old_graph_node
                                                            find_same = True

                                                    # Only execute if no matching coalition was found
                                                if not find_same:
                                                        # Create new node
                                                        new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                        # Store in state map (dictionary)
                                                        self.state_map[str(new_graph_coalitionx)] = new_node

                                                find_same_child = False
                                                    # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                for cur_child in tree_node.children:
                                                        if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                            find_same_child = True

                                                if not find_same_child:
                                                        tree_node.children.append(new_node)
                                                        # ycounter+=1
                                                # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                                # The tree node's children contribution (P) is then computed using Shapley value function
                                                scores = compute_scores(self.score_func, tree_node.children)
                                                for child, score in zip(tree_node.children, scores):
                                                    child.P = score
                                                    # ycounter =0 # reset child count
                                
                                
                            elif len(ordered_nodesA) > BaseNum :
                            # L2 block
                                # case of nodes in the adjacent set being more than number required for base nodes
                                
                                                           
                                    
                                if (len(ordered_nodesA) - BaseNum) < config.NumOfCombinationSets :
                                # L3 block
                                    # If the remaining adjacent node set after forming the base is less then the combinatorial required number
                                    # then use the remaining non-adjacent node set after exhausting the remaining adjacent set
                                    # first_list is the base node set
                                    # second_list is the remaining set of adjacent nodes
                                    
                                    AdjacentNodesx = ordered_nodesA
                                                                
                                    first_list, second_list = split_node_list_ordered(AdjacentNodesx, BaseNum)
                                    nodes_to_process = list(second_list)[:config.NumOfCombinationSets] # Note this set will not be enough
                                    BaseNodes = first_list
                                    for node in nodes_to_process:
                                   
                                        subgraph_coalitionz = [self.node_idx] + list(BaseNodes) + [node]

                                        remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                        subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                              
                                        #   print(f"Adjacent nodes coalition: {subgraph_coalitionz}") 
                                        # Generate connected subgraphs from the coalition
                                        subgraphsx = [self.graph.subgraph(c)
                                                        for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                
                                                                          
                                        main_subx = subgraphsx[0]
                                      
                                        # If a 'node_idx' exists, find the subgraph containing it
                                        if self.node_idx != None :
                                                for subx in subgraphsx:
                                                    if self.node_idx in list(subx.nodes()):
                                                        main_subx = subx
                                                        
                                        else:
                                                for subx in subgraphsx:
                                                    if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                        main_subx = subx
                                                        
                                                    

                                        new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                            # check the state map and merge the same sub-graph
                                        find_same = False
                                            # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                            # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                            # this checks all MCTS nodes created so far fr equality
                                        for old_graph_node in self.state_map.values():
                                                if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                    new_node = old_graph_node
                                                    find_same = True

                                            # Only execute if no matching coalition was found
                                        if not find_same:
                                                # Create new node
                                                new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                # Store in state map (dictionary)
                                                self.state_map[str(new_graph_coalitionx)] = new_node

                                        find_same_child = False
                                            # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                        for cur_child in tree_node.children:
                                                if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                    find_same_child = True

                                        if not find_same_child:
                                                tree_node.children.append(new_node)
                                                # ycounter+=1
                                    
                                    # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                    # The tree node's children contribution (P) is then computed using Shapley value function
                                    scores = compute_scores(self.score_func, tree_node.children)
                                    for child, score in zip(tree_node.children, scores):
                                            child.P = score
                                            # ycounter =0 # reset child count

                                
                                    # This time you are splitting the non-adjacent nodes
                                    if len(ordered_nodes) <= (config.NumOfCombinationSets-(len(ordered_nodesA) - BaseNum)):
                                                first_list, second_list = split_node_list_ordered(ordered_nodes, len(ordered_nodes))
                                                nodes_to_process = list(first_list)[:len(ordered_nodes)] 
                                                for node in nodes_to_process:
   
                                                    subgraph_coalitionz = [self.node_idx] + list(BaseNodes) + [node]
                                                    # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                    # Generate connected subgraphs from the coalition

                                                    remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                    subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                                                    subgraphsx = [self.graph.subgraph(c)
                                                                for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                        
                                                    main_subx = subgraphsx[0]
                                                                                               
                                                   
                                                    # If a 'node_idx' exists, find the subgraph containing it
                                                    if self.node_idx != None :
                                                            for subx in subgraphsx:
                                                                if self.node_idx in list(subx.nodes()):
                                                                    main_subx = subx
                                                                
                                                    else:
                                                            for subx in subgraphsx:
                                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                    main_subx = subx
                                                                
                                                                    

                                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                    # check the state map and merge the same sub-graph
                                                    find_same = False
                                                    # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                    # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                    # this checks all MCTS nodes created so far fr equality
                                                    for old_graph_node in self.state_map.values():
                                                        if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                            new_node = old_graph_node
                                                            find_same = True

                                                    # Only execute if no matching coalition was found
                                                    if not find_same:
                                                        # Create new node
                                                        new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                        # Store in state map (dictionary)
                                                        self.state_map[str(new_graph_coalitionx)] = new_node

                                                    find_same_child = False
                                                    # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                    for cur_child in tree_node.children:
                                                        if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                            find_same_child = True

                                                    if not find_same_child:
                                                        tree_node.children.append(new_node)
                                                        # ycounter+=1
                                                # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                                # The tree node's children contribution (P) is then computed using Shapley value function
                                                scores = compute_scores(self.score_func, tree_node.children)
                                                for child, score in zip(tree_node.children, scores):
                                                        child.P = score
                                                        # ycounter =0 # reset child count

                                    elif len(ordered_nodes) > (config.NumOfCombinationSets-(len(ordered_nodesA) - BaseNum)):
                                               
                                                AdjacentNodesx = ordered_nodesA
                                                NonAdjacentNodesx = ordered_nodes
                                                first_list, second_list = split_node_list_ordered(NonAdjacentNodesx, config.NumOfCombinationSets-(len(AdjacentNodesx) - BaseNum))
                                                nodes_to_process = list(first_list)[:config.NumOfCombinationSets -(len(AdjacentNodesx) - BaseNum)] # Note this set will not be enough
                                                for node in nodes_to_process:
                                                    
                                                                                                    
                                                    subgraph_coalitionz = [self.node_idx] + list(BaseNodes) + [node]

                                                    remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                                    subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                                                    # print(f"Non-adjacent nodes coalition: {subgraph_coalitionz}") 
                                                    # Generate connected subgraphs from the coalition
                                                    subgraphsx = [self.graph.subgraph(c)
                                                                    for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                            
                                                    main_subx = subgraphsx[0]
                                                  
                                                    # If a 'node_idx' exists, find the subgraph containing it
                                                    if self.node_idx != None :
                                                            for subx in subgraphsx:
                                                                if self.node_idx in list(subx.nodes()):
                                                                    main_subx = subx
                                                                
                                                    else:
                                                            for subx in subgraphsx:
                                                                if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                                    main_subx = subx
                                                                
                                                                

                                                    new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                                        # check the state map and merge the same sub-graph
                                                    find_same = False
                                                        # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                                        # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                                        # this checks all MCTS nodes created so far fr equality
                                                    for old_graph_node in self.state_map.values():
                                                            if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                                new_node = old_graph_node
                                                                find_same = True

                                                        # Only execute if no matching coalition was found
                                                    if not find_same:
                                                            # Create new node
                                                            new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                            # Store in state map (dictionary)
                                                            self.state_map[str(new_graph_coalitionx)] = new_node

                                                    find_same_child = False
                                                        # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                                    for cur_child in tree_node.children:
                                                            if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                                find_same_child = True

                                                    if not find_same_child:
                                                            tree_node.children.append(new_node)
                                                            # ycounter+=1
                                                
                                                
                                                # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                                                # The tree node's children contribution (P) is then computed using Shapley value function
                                                scores = compute_scores(self.score_func, tree_node.children)
                                                for child, score in zip(tree_node.children, scores):
                                                        child.P = score
                                                        # ycounter =0 # reset child count
                                
                                                                    
                                elif len(ordered_nodesA) - BaseNum  >= config.NumOfCombinationSets :
                                # L3 block
                                    # first_list is the base node set
                                    # second_list is the remaining set of adjacent nodes
                                    AdjacentNodesx = ordered_nodesA
                                    first_list, second_list = split_node_list_ordered(AdjacentNodesx, BaseNum)
                                    nodes_to_process = list(second_list)[:config.NumOfCombinationSets]
                                    for node in nodes_to_process:
                                        
                                                                              
                                        subgraph_coalitionz = [self.node_idx] + list(first_list) + [node]
                                        # Generate connected subgraphs from the coalition

                                        remnamt = [item for item in ordered_nodes if item not in subgraph_coalitionz]
                                        subgraph_coalitionz = list(subgraph_coalitionz)+list(remnamt)

                                        subgraphsx = [self.graph.subgraph(c)
                                                        for c in nx.connected_components(self.graph.subgraph(subgraph_coalitionz))]
                                                                                
                                        main_subx = subgraphsx[0]
                                                                                               
                                      
                                        # If a 'node_idx' exists, find the subgraph containing it
                                        if self.node_idx != None :
                                                for subx in subgraphsx:
                                                    if self.node_idx in list(subx.nodes()):
                                                        main_subx = subx
                                                        
                                        else:
                                                for subx in subgraphsx:
                                                    if subx.number_of_nodes() > main_subx.number_of_nodes():
                                                        main_subx = subx
                                                        
                                                    

                                        new_graph_coalitionx = sorted(list(main_subx.nodes()))
                                            # check the state map and merge the same sub-graph
                                        find_same = False
                                            # Loops over all previously explored nodes (old_graph_node) stored in self.state_map
                                            # self.state_map is typically a dictionary mapping node IDs to their historical data.
                                            # this checks all MCTS nodes created so far fr equality
                                        for old_graph_node in self.state_map.values():
                                                if Counter(old_graph_node.coalition) == Counter(new_graph_coalitionx):
                                                    new_node = old_graph_node
                                                    find_same = True

                                            # Only execute if no matching coalition was found
                                        if not find_same:
                                                # Create new node
                                                new_node = self.MCTSNodeClass(new_graph_coalitionx)
                                                # Store in state map (dictionary)
                                                self.state_map[str(new_graph_coalitionx)] = new_node

                                        find_same_child = False
                                            # Loops over all existing child nodes (cur_child) of the current MCTS tree node (tree_node.children)
                                        for cur_child in tree_node.children:
                                                if Counter(cur_child.coalition) == Counter(new_graph_coalitionx):
                                                    find_same_child = True

                                        if not find_same_child:
                                                tree_node.children.append(new_node)
                                                # ycounter+=1
                            
                            # print(f"Number of children is : {ycounter} for level/depth {xcounter} with coalition number of : {len(expand_nodes)}")            # At this point, the tree node and its corresponding child nodes have been created in MCTS tree
                            # The tree node's children contribution (P) is then computed using Shapley value function
                            scores = compute_scores(self.score_func, tree_node.children)
                            for child, score in zip(tree_node.children, scores):
                                        child.P = score
                                        # ycounter =0 # reset child count

                

            # --------------------------------      
                    
            elif config.DargumentX == 'MCTS' :   
                node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
                node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=mcts_args.high2low)
                all_nodes = [x[0] for x in node_degree_list]

                if len(all_nodes) < self.expand_atoms:
                    expand_nodes = all_nodes
                else:
                    expand_nodes = all_nodes[:self.expand_atoms]

                for each_node in expand_nodes:
                    # for each node, pruning it and get the remaining sub-graph
                    # here we check the resulting sub-graphs and only keep the largest one
                    subgraph_coalition = [node for node in all_nodes if node != each_node]

                    subgraphs = [self.graph.subgraph(c)
                                for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]
                    main_sub = subgraphs[0]
                    for sub in subgraphs:
                        if sub.number_of_nodes() > main_sub.number_of_nodes():
                            main_sub = sub

                    new_graph_coalition = sorted(list(main_sub.nodes()))

                    # check the state map and merge the same sub-graph
                    Find_same = False
                    for old_graph_node in self.state_map.values():
                        if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                            new_node = old_graph_node
                            Find_same = True

                    if Find_same == False:
                        new_node = self.MCTSNodeClass(new_graph_coalition)
                        self.state_map[str(new_graph_coalition)] = new_node

                    Find_same_child = False
                    for cur_child in tree_node.children:
                        if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                            Find_same_child = True

                    if Find_same_child == False:
                        tree_node.children.append(new_node)

                scores = compute_scores(self.score_func, tree_node.children)
                for child, score in zip(tree_node.children, scores):
                    child.P = score

        else:
         print("Return the computed reward ...")
        sum_count = sum([c.N for c in tree_node.children])
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):
        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"At the {rollout_idx} rollout, {len(self.state_map)} states that have been explored.")

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data)
        else:
            score = child.P
        results.append(score)
    return results


def reward_func(reward_args, value_func):
    if reward_args.reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method)

    elif reward_args.reward_method.lower() == 'mc_shapley':
        return partial(mc_shapley,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method,
                       sample_num=reward_args.sample_num)

    elif reward_args.reward_method.lower() == 'l_shapley':
        return partial(l_shapley,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method)

    elif reward_args.reward_method.lower() == 'mc_l_shapley':
        return partial(mc_l_shapley,
                       local_raduis=reward_args.local_raduis,
                       value_func=value_func,
                       subgraph_building_method=reward_args.subgraph_building_method,
                       sample_num=reward_args.sample_num)
    else:
        raise NotImplementedError

