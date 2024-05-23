# Script to run on server
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
import pickle

random.seed(1) # use a seed to get the same values 

# Functions I need:
def add_or_increase_edge(graph, source, target, w0):
    # Check if an edge exists between source and target
    if graph.has_edge(source, target):
        # If an edge exists, increase the weight
        graph[source][target]['weight'] += w0
    else:
        # If no edge exists, add a new edge
        graph.add_edge(source, target, weight=w0)

def generate_base_graph_random_fitness(n0, w0, eta_min, eta_max):
    # Create a directed graph with n0 nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n0))

    # Add edges between every pair of nodes with weight w0
    for source in range(n0):
        for target in range(n0):
            if source != target:  # Avoid self-loops
                G.add_edge(source, target, weight=w0)

    # Assign random in_fitness and out_fitness to each node
    in_fitnesses = [random.randint(eta_min, eta_max) for _ in range(n0)]
    out_fitnesses = [random.randint(eta_min, eta_max) for _ in range(n0)]
   
    nx.set_node_attributes(G, dict(enumerate(in_fitnesses)), name="in_fitness")
    nx.set_node_attributes(G, dict(enumerate(out_fitnesses)), name="out_fitness")

    return G

# generate a base_graph and add the roles to the nodes, r values represent the proportion of different roles
def generate_base_graph_fixed_fitness_roles(n0, w0, r0, r1, r2):
    # Create a directed graph with n0 nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n0))

    # Add edges between every pair of nodes with weight w0
    for source in range(n0):
        for target in range(n0):
            if source != target:  # Avoid self-loops
                G.add_edge(source, target, weight=w0)

    # Assign random in_fitness and out_fitness to each node
    in_fitnesses = [1 for _ in range(n0)]
    out_fitnesses = [3 for _ in range(n0)]

    # Assign roles to each node based on the provided probabilities
    roles = []
    for _ in range(n0):
        role = random.choices([0, 1, 2], weights=[r0, r1, r2], k=1)[0]
        roles.append(role)
   
    nx.set_node_attributes(G, dict(enumerate(in_fitnesses)), name="in_fitness")
    nx.set_node_attributes(G, dict(enumerate(out_fitnesses)), name="out_fitness")
    nx.set_node_attributes(G, dict(enumerate(roles)), name="role")

    return G

# I fot this code from nx
def _random_subset(seq, m, rng):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        if not seq:
            print("empty seq")
            break  # Exit the loop if seq is empty
        x = rng.choice(seq)
        targets.add(x)
    return targets

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

def plot_communities(community, input_label):
    # Calculate the frequency of each community
    com_freq = {}
    total_nodes = 0

    for i, com in enumerate(community):
        com_size = len(com)
        total_nodes += com_size
        com_freq[i] = com_size

    # Calculate the fraction of nodes in each community
    frac = [freq / total_nodes for freq in com_freq.values()]

    # Plot the frequency against the fraction
    plt.scatter(frac, com_freq.values(), label=input_label)
    plt.xlabel('Fraction of Nodes in Community')
    plt.ylabel('Frequency of Community')
    plt.legend()

def plot_community_sizes(community, input_label):
    # Count the number of communities and calculate community sizes
    num_communities = len(community)
    community_sizes = [len(com) for com in community]

    # Plot the number of communities versus community sizes
    plt.bar(range(num_communities), community_sizes, label=input_label)
    plt.xlabel('Community Index')
    plt.ylabel('Community Size')
    plt.legend()

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

def metric_values(G, weight_param):
    graph_edge_weights = G.es[weight_param]

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

    print(G_weighted_CC.is_simple())
    # CC_edge_weights = G_weighted_CC.es[weight_param]
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
    # dom_eigenvec = dominant_eigenvector(adjacency_matrix)[0]

    eigenvec = dominant_eigenvector(adjacency_mat)[1]
    sp_radius_eigenvec = dominant_eigenvector(adjacency_mat)[0]
    var_eigenvec = np.var(eigenvec)
    skew_eigenvec = sp.stats.skew(eigenvec)
    
    # community detection
    # extra_weights = graph_edge_weights.copy()
    # G_copy = G.copy()
    
    leiden_com = G.as_undirected().community_leiden(weights=graph_edge_weights)
    infomap_com = G.community_infomap(edge_weights=graph_edge_weights)
    # plot_community_sizes(leiden_com, 'leiden')
    # plot_communities(infomap_com, 'infomap')
    
    # core periphery
    in_coreness = G.coreness(mode='in')
    out_coreness = G.coreness(mode='out')
    coreness = G.coreness(mode='all')
    in_shell_index_counts = Counter(in_coreness)
    out_shell_index_counts = Counter(out_coreness)
    shell_index_counts = Counter(coreness)

    # graph density
    density = G.density(loops=False)
    
    return f"""
In-strength stats: 
  Median: {median_in_strength}
  Standard Deviation: {std_in_strength}
  Maximum: {max_in_strength}
  Minimum: {min_in_strength}

Out-strength stats:
  Median: {median_out_strength}
  Standard Deviation: {std_out_strength}
  Maximum: {max_out_strength}
  Minimum: {min_out_strength}

Clustering:
  CC: {CC}
  Weighted CC: {weighted_CC}

Assortativity:
  Degree Assortativity: {degree_ass}
  Strength Assortativity: {strength_ass}

Community structures: 
  Leiden community: {leiden_com.summary()}
  Infomap community: {infomap_com.summary()}

Core periphery structure:
  In-coreness: {in_shell_index_counts}
  Out-coreness: {out_shell_index_counts}
  Coreness: {shell_index_counts}

Power law exponents:
  In-degree: {in_degree_p_law_exp.alpha}
  Out-degree: {out_degree_p_law_exp.alpha}
  In-strength: {in_str_p_law_exp.alpha}
  Out-strength: {out_str_p_law_exp.alpha}

Eigenvector metrics:
  Spectral radius: {sp_radius_eigenvec}
  Variance dom. eigenvec: {var_eigenvec}
  Skewness dom. eigenvec: {skew_eigenvec}

Density:
  Edge density: {density}

""" 

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

# model without fitness, but with roles
def model_f_roles(G, p, w0, num_iter, dens_param_in, dens_param_out, pr_mat, pr_f, new_node_m):
    """
    G = the graph, nx object
    p = proba of generating an edge
    w0 = weight on edges
    num_iter = number of iterations the model will run
    dense_param_in, dense_param_out = the number of incoming and outgoing edges that will be added in the densification step
    tr_mat = transition matrix for probability of edges, based on node-type
    pr_f = probability function of which node-type will be generated {0: ..., 1:..., 2:...}
    new_node_m = the number of incoming and outgoing edges for each role at birth

    """
    # keep a list of the edges (in/out) per node, multiply with fitness
    rep_in_c0 = [n for n, d in G.in_degree() if G.nodes[n]["role"] == 0 for _ in range(d)]
    rep_out_c0 = [n for n, d in G.out_degree() if G.nodes[n]["role"] == 0 for _ in range(d)]
    rep_in_c1 = [n for n, d in G.in_degree() if G.nodes[n]["role"] == 1 for _ in range(d)]
    rep_out_c1 = [n for n, d in G.out_degree() if G.nodes[n]["role"] == 1 for _ in range(d)]
    rep_in_c2 = [n for n, d in G.in_degree() if G.nodes[n]["role"] == 2 for _ in range(d)]
    rep_out_c2 = [n for n, d in G.out_degree() if G.nodes[n]["role"] == 2 for _ in range(d)]

    # Concat them:
    rep_in = [rep_in_c0, rep_in_c1, rep_in_c2]
    rep_out = [rep_out_c0, rep_out_c1, rep_out_c2]

    for _ in tqdm(range(num_iter)):
        # pick random number for proba
        random_number = random.random()
        source = len(G) # source node index
        
        # with probability p, network growth
        if random_number <= p: 
            # print("new node joining") # for debugging
            # choose a random role for the new node using the probability function
            role = random.choices(list(pr_f.keys()), weights=list(pr_f.values()))[0] 
            
            for _ in range(new_node_m[role]["m_in"]):
                # Calculate the probability of outgoing edges for the current node's role
                out_role_p = pr_mat[:, role]
                out_role_p_c = out_role_p.copy()

                # Normalize the probabilities
                out_role_p_c /= np.sum(out_role_p_c)
                available_roles = list(range(len(out_role_p_c)))

                # Loop until a valid role for outgoing edges is found
                while True:
                    # Choose a role based on the normalized probabilities
                    out_role = np.random.choice(available_roles, p=out_role_p_c[available_roles])
                    # Check if the chosen role has nodes available for outgoing edges
                    if rep_out[out_role]:
                        break

                    # If not, remove the chosen role from the available roles and check for more options
                    available_roles.remove(out_role)
                    # If no available roles remain or all probabilities are zero, exit the loop
                    if not available_roles or all(el == 0 for el in out_role_p_c[available_roles]):
                        break
                    # Normalize the probabilities again based on available roles
                    out_role_p_c[available_roles] /= np.sum(out_role_p_c[available_roles])

                # If there are nodes available for outgoing edges for the chosen role
                if rep_out[out_role]:
                    # Select a random node from the available nodes for outgoing edges
                    in_target = _random_subset(rep_out[out_role], 1, np.random.default_rng(seed=1))
                    # Add an edge from the selected node to the current node with the specified weight
                    if in_target:
                        G.add_edge(list(in_target)[0], source, weight=w0)
                        # Update the array of repeated outgoing nodes for the chosen role
                        rep_out[out_role].extend(in_target)


            for _ in range(new_node_m[role]["m_out"]):
                # Calculate the probability of incoming edges for the current node's role
                in_role_p = pr_mat[role, :]
                in_role_p_c = in_role_p.copy()

                # Normalize the probabilities
                in_role_p_c /= np.sum(in_role_p_c)
                available_roles = list(range(len(in_role_p_c)))

                # Loop until a valid role for incoming edges is found
                while True:
                    # Choose a role based on the normalized probabilities
                    in_role = np.random.choice(available_roles, p=in_role_p_c[available_roles])
                    # Check if the chosen role has nodes available for incoming edges
                    if rep_in[in_role]:
                        break
                    # If not, remove the chosen role from the available roles and check for more options
                    available_roles.remove(in_role)
                    # If no available roles remain or all probabilities are zero, exit the loop
                    if not available_roles or all(el == 0 for el in in_role_p_c[available_roles]):
                        break
                    # Normalize the probabilities again based on available roles
                    in_role_p_c[available_roles] /= np.sum(in_role_p_c[available_roles])

                # If there are nodes available for incoming edges for the chosen role
                if rep_in[in_role]:
                    # Select a random node from the available nodes for incoming edges
                    out_target = _random_subset(rep_in[in_role], 1, np.random.default_rng(seed=1))
                    # Add an edge from the current node to the selected node with the specified weight
                    if out_target:
                        G.add_edge(source, list(out_target)[0], weight=w0)
                        # Update the array of repeated incoming nodes for the chosen role
                        rep_in[in_role].extend(out_target)


            if G.has_node(source): # if the node is created, update the fitness parameters
                # give fitness value to the newly created node -> each role has different fitness values
                G.nodes[source]['role'] = role

                # update the array for the newly created node role   
                if new_node_m[role]["m_in"] != 0: # check whether m_in was eq to 0 for the new node, if not, add to repeated in nodes
                    rep_in[role].extend([source] * new_node_m[role]["m_in"])
                if new_node_m[role]["m_out"] != 0: # check whether m_out was eq to 0 for the new node, if not, add to repeated out nodes
                    rep_out[role].extend([source] * new_node_m[role]["m_out"])
 
        # densification
        else: 
            # print("densification") # use for debugging

            # Pick random nodes for increasing an edge weight or adding an edge -> Should this be random???
            in_targets = random.sample([node for node, attr in G.nodes(data=True) if attr['role'] in [1, 2]], dens_param_in) # only allowed to take a role 1 or 2, because 0 cannot get incoming edges
            out_sources = random.sample([node for node, attr in G.nodes(data=True) if attr['role'] in [0, 2]], dens_param_out) # only take role 0 or 2

            for in_t in in_targets:
                # the role of the chosen node (random)
                role = G.nodes[in_t]['role'] 

                # find the role of node it should connect to
                out_role_p = pr_mat[:, role] 

                out_role_p_c = out_role_p.copy() # copy otherwise things mess up
                out_role_p_c /= np.sum(out_role_p_c) # normalize 
                in_s_role = np.random.choice(len(out_role_p), p=out_role_p_c) # choose the role that will connect to in_t

                available_roles = list(range(len(out_role_p_c)))
                    
                # take an out-role set, but not an empty one
                while not rep_out[in_s_role]:
                    # Exclude the chosen role from the options
                    available_roles.remove(in_s_role)
                    if not available_roles or all(el == 0 for el in out_role_p_c[available_roles]) :
                        break  # Exit the loop if no roles are available                        
               
                    out_role_p_c[available_roles] /= np.sum(out_role_p_c[available_roles])
                    in_s_role = np.random.choice(available_roles, p=out_role_p_c[available_roles])
                
                # check again, if it doesn't exist, exit    
                if not rep_out[in_s_role]:
                    break              

                # pick the in source node (a node which will have the outgoing edges)
                in_s = _random_subset(rep_out[in_s_role], 1, np.random.default_rng(seed=1))

                # check whether we are creating a self-loop or not, if we do, we pick another random in_source node (with the same role)
                # maybe need to add a check to see whether the sets have at least 2 elems if we pick the same role, otherwise we can still generate a self-loop?
                while in_s == in_t:
                    nodes_with_role = [node for node, data in G.nodes(data=True) if data['role'] == role]
                    in_t = random.choice(nodes_with_role)

                if in_s:
                    source = list(in_s)[0]
                    if G.has_edge(source, in_t):
                        # If the edge already exists, update its weight
                        G[source][in_t]['weight'] += w0
                    else:
                        # If the edge doesn't exist, add it with the specified weight
                        G.add_edge(source, in_t, weight=w0)
                    
                    # update the arrays
                    rep_out[in_s_role].extend([source])
                    rep_in[role].extend([in_t])
                            
            for out_s in out_sources:
                # the role of the chosen node (random)
                role = G.nodes[out_s]['role'] 

                # find the role of node it should connect to
                in_role_p = pr_mat[role, :] 
                in_role_p_c = in_role_p.copy()
                in_role_p_c /= np.sum(in_role_p_c) # normalize (maybe not necessary)
                out_t_role = np.random.choice(len(in_role_p), p=in_role_p_c) # choose the role that will connect to out_s

                available_roles = list(range(len(out_role_p_c)))
                    
                # take an out-role set, but not an empty one
                while not rep_in[out_t_role]:

                    # Exclude the chosen role from the options
                    available_roles.remove(out_t_role)
                    if not available_roles or all(el == 0 for el in in_role_p_c[available_roles]) :
                        break  # Exit the loop if no roles are available                        
               
                    in_role_p_c[available_roles] /= np.sum(in_role_p_c[available_roles])
                    out_t_role = np.random.choice(available_roles, p=in_role_p_c[available_roles])
                    
                if not rep_out[out_t_role]:
                    break   

                # pick the in source node (a node which will have the outgoing edges)
                out_t = _random_subset(rep_out[out_t_role], 1, np.random.default_rng(seed=1))

                # check whether we are creating a self-loop or not, if we do, we pick another random in_source node (with the same role)
                # maybe need to add a check to see whether the sets have at least 2 elems if we pick the same role, otherwise we can still generate a self-loop!!!
                while out_t == out_s:
                    nodes_with_role = [node for node, data in G.nodes(data=True) if data['role'] == role]
                    out_s = random.choice(nodes_with_role)

                if out_t:
                    target = list(out_t)[0]
                    if G.has_edge(out_s, target):
                        # If the edge already exists, update its weight
                        G[out_s][target]['weight'] += w0
                    else:
                        # If the edge doesn't exist, add it with the specified weight
                        G.add_edge(out_s, target, weight=w0)
                    
                    # update the arrays
                    rep_out[out_t_role].extend([out_s])
                    rep_in[role].extend([target])
        
    return(G)

# model paramters:
new_nodes = {
    0: {"m_in":0 , "m_out":5},
    1: {"m_in": 5, "m_out":0},
    2: {"m_in": 5, "m_out": 5}
}

Gr = generate_base_graph_fixed_fitness_roles(5, 1, 0, 0, 1) # we use a basegraph with only role 2 nodes, and fully connected
pr_mat_t = np.array([[0, 8624, 1224003],[0, 0, 0],[0, 457620, 2435032]]).astype(float) # probability matrix similar to that of Rabo
pr_f_t = {0: 0.6, 1: 0.2, 2: 0.2} # probability similar to Rabo outcome


# Run the model with the paramters:
Gr = model_f_roles(Gr, p=0.8, w0=1, num_iter=1000000, dens_param_in=5, dens_param_out=3, pr_mat=pr_mat_t, pr_f=pr_f_t, new_node_m=new_nodes)

# Save the created graph
with open('graph_1mil_iter_tuned.pkl', 'wb') as f:
    pickle.dump(Gr, f)


