# This code can be ran to analyze the metrics of a generated graph
from igraph import Graph
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import statistics
import powerlaw
import scipy as sp
from itertools import product
import re
from tqdm import tqdm
import random
from collections import Counter

#function to find strength stats, can be in- or out- based on param, and weight specified by weight_param
def find_strength(G, param, weight_param):
    strengthlist = G.strength(mode=param, weights=weight_param)
    median = statistics.median(strengthlist)
    std = statistics.stdev(strengthlist)
    max_str = max(strengthlist)
    min_str = min(strengthlist)
    return(median, std, max_str, min_str, strengthlist)

# for clustering, otherwise doesn't work
def aggregate_multi_edges(graph):
    aggregated_graph = graph.copy()

    # Collapse multi-edges by aggregating their weights
    aggregated_graph.simplify(combine_edges='sum')
    return aggregated_graph

def dominant_eigenvector(adjacency_matrix, num_iterations=1000, tolerance=1e-6):
    n = len(adjacency_matrix)

    # Convert igraph Matrix to NumPy array
    adjacency_matrix = np.array(adjacency_matrix)

    # Initialize a random vector
    x = np.random.rand(n)
    x /= np.linalg.norm(x)

    for _ in range(num_iterations):
        # Multiply by the adjacency matrix
        Ax = np.dot(adjacency_matrix, x)

        # Compute the dominant eigenvalue
        eigenvalue = np.linalg.norm(Ax)

        # Normalize the vector
        x = Ax / eigenvalue

        # Check for convergence
        if np.linalg.norm(Ax - eigenvalue * x) < tolerance:
            break

    return eigenvalue, x

def generate_igraph(G):   
    # Create an iGraph graph from the networkX graph
    igraph_graph = Graph(directed=True)

    # Add vertices to the iGraph graph
    igraph_graph.add_vertices(list(G.nodes))

    # Add edges to the iGraph graph along with their weights
    for edge in G.edges(data=True):
        source, target, weight = edge
        igraph_graph.add_edge(source, target, weight=weight['weight'])
    return(igraph_graph)

def f_mean_metric_values(G, weight_param):
    # graph_edge_weights = G.es[weight_param]

    # in- and out-strength stats
    median_in_strength, std_in_strength, max_in_strength, min_in_strength, in_str = find_strength(G, "in", weight_param) 
    median_out_strength, std_out_strength, max_out_strength, min_out_strength, out_str = find_strength(G, "out", weight_param) 
    
    # weighted and unweighted CC
    G_CC = G.copy()
    CC_list = aggregate_multi_edges(G_CC).transitivity_local_undirected(mode='zero', weights=None)
    CC = statistics.mean(CC_list)
    
    G_weighted_CC = G.copy()
    G_weighted_CC.as_undirected()
    G_weighted_CC = aggregate_multi_edges(G_weighted_CC)
    weighted_CC_list = G_weighted_CC.transitivity_local_undirected(mode='zero', weights=None)
    weighted_CC = statistics.mean(weighted_CC_list)
    
    # degree and strength assortativity 
    degree_ass = G.assortativity_degree(directed=True)
    strength_ass = G.assortativity(types1=out_str, types2=in_str, directed=True)
    
    # power law exp
    in_degrees = G.indegree()
    out_degrees = G.outdegree()
    in_degree_p_law_exp = powerlaw.Fit(in_degrees, verbose=False)
    out_degree_p_law_exp = powerlaw.Fit(out_degrees, verbose=False)
    in_str_p_law_exp = powerlaw.Fit(in_str, verbose=False)
    out_str_p_law_exp = powerlaw.Fit(out_str, verbose=False)
    
    # eigenvector metrics
    adjacency_mat = G.get_adjacency().data
    eigenvec = dominant_eigenvector(adjacency_mat)[1]
    sp_radius_eigenvec = dominant_eigenvector(adjacency_mat)[0]
    var_eigenvec = np.var(eigenvec)
    skew_eigenvec = sp.stats.skew(eigenvec)
    
    # graph density
    density = G.density(loops=False)
    
    return median_in_strength, std_in_strength, max_in_strength, min_in_strength, \
           median_out_strength, std_out_strength, max_out_strength, min_out_strength, \
           CC, weighted_CC, degree_ass, strength_ass, in_degree_p_law_exp.alpha, \
           out_degree_p_law_exp.alpha, in_str_p_law_exp.alpha, out_str_p_law_exp.alpha, \
           sp_radius_eigenvec, var_eigenvec, skew_eigenvec, density


# Take one of the generated graphs, comment in below and change name
Gr = nx.read_gpickle('graph_500k_iter.gpickle') 
Gr_igr = generate_igraph(Gr) # create an igraph object

all_metric_values = [[] for _ in range(20)]  # Number of metrics = 20

column_names = [
    'm_out', 'm_in', 'Fitness_Matrix', 'p', 'Median_In_Strength', 'StdDev_In_Strength', 'Max_In_Strength', 'Min_In_Strength',
    'Median_Out_Strength', 'StdDev_Out_Strength', 'Max_Out_Strength', 'Min_Out_Strength', 'CC', 'Weighted_CC',
    'Degree_Assortativity', 'Strength_Assortativity', 'Power_Law_Exponent_In_Degree', 'Power_Law_Exponent_Out_Degree',
    'Power_Law_Exponent_In_Strength', 'Power_Law_Exponent_Out_Strength', 'Spectral_Radius', 'Variance_Dom_Eigenvec',
    'Skewness_Dom_Eigenvec', 'Edge_Density'
]
data = [] # Create a list to store the parameter combinations and metric values
metric_output = list(f_mean_metric_values(Gr_igr, 'total'))  
data.append(metric_output)
df = pd.DataFrame(data, columns=column_names) # Create df with correct column labels
df.to_csv('metric_outcomes_rabo_unweighted', index=False) # save the values as a csv
