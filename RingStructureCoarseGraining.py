# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#---------------------------------------------------- COARSE GRAINING WITH RING STRUCTURE ------------------------------------------------------------------
""" Based on "Ring structures and mean first passage time in networks"
    by Andrea Baronchelli and Vittorio Loreto
"""

import networkx as nx
import numpy as np
import scipy.sparse as ss
import random


def __main__():
    g = nx.read_dot('shortened_winter_wheat_56mresolution_upstate_ny_20box_radius_inf.dot')
    #g = nx.read_dot('ExampleGraph1.dot')
    cc = nx.connected_component_subgraphs(g)
    for sub_graph in cc:
        if len(sub_graph.nodes()) > 2:
            A = nx.adjacency_matrix(sub_graph)
            MFPTs = []
            for node in sub_graph.nodes():
                MFPTs.append(find_analytical_ring_MFPT(node, sub_graph, 0.001, 500))
            fout = open('ring_MFPT_connected_component' + str(cc.index(sub_graph)) + '.npy', 'w+')
            np.save(fout, MFPTs)
            fout.close
                 
    return None


def find_analytical_ring_MFPT(start_node, g, tolerance, max_iteration):
    """ Input: start_node in 'x y' format. g = nx.graph. Contains infinite
        sum, so tolerance and max interation are included. A tolerance of 0.01
        produced satisfactory convergence for me. (Although this obviously depended
        on the trial graph I was using)
    """
                
    rings = make_rings(g, start_node)
    diag_m = find_diagonal_m(rings, g)
    off_diag_m = find_off_diagonal_m(rings, g)
    B = make_B(diag_m, off_diag_m)
    
    term = 1.0
    error = 0.0
    total = 0.0
    counter = 0
    time = 1
    
    while term > tolerance and counter < max_iteration:
        term = time*find_first_passage_probability(time, rings, B, g)
        total += term
        time += 1
        counter += 1
        
    return total



def make_rings(g, start_node):
    """ Find the rings around a given node, composing a graph. A ring of length l
        centered as node i, is defined as the set of nodes that are 'l' edges away
        from the node i.

        Input: nx.Graph, and a node in this graph.
        Output: a dictionary, {L, ring of length L}.
    """

    rings = {}

    for i in g.nodes():
        length = nx.shortest_path_length(g, start_node, i)
        
        #If ring hasn't been created already, do so
        if length not in rings.keys():
            rings[length] = []
        
        #add node to relevant ring
        rings[length].append(i)
        
    return rings



def find_diagonal_m(rings, g):
    """ Finds the diagonal element of the matrix 'm',
        as defined in the coarse graining procedure.
        (An intermediary to the matrix B)

        Input: nx.Graph, and dictionary of {L, ring of length L}
    """
    
    diag_m = []
    
    for ring in rings.values():
        degree_distribution_per_ring = []
        
        for node in ring:
            internal_degree_node_i = 0.0
            
            for neighbour in g[node]:     #all of node's neighbours
                if neighbour in ring:
                    internal_degree_node_i += 1
                
            degree_distribution_per_ring.append(internal_degree_node_i)
            
        diag_m.append(sum(degree_distribution_per_ring))
            
    return diag_m



def find_off_diagonal_m(rings, g):
    """ Intermediary Function to find matrix B """
    
    off_diag_m = []
    
    for k in range(len(rings)-1):
        ring1 = rings.values()[k]
        ring2 = rings.values()[k+1]
        degree_distribution_per_ring = []
        
        for node in ring1:
            external_degree_node_i = 0.0
            
            for neighbour in g[node]:
                if neighbour in ring2:
                    external_degree_node_i += 1
            
            degree_distribution_per_ring.append(external_degree_node_i)
        
        off_diag_m.append(sum(degree_distribution_per_ring))
        
    return off_diag_m



def make_B(diag_m, off_diag_m):
    """Constructs the matrix B, defined in paper, which represents
       the transition matrix for the ring process: B_ij(t) = prob.
       to go from ring_i -> ring_j in t steps. We are interested in 
       FIRST passage time, so this is an absorbing random walk, so
       the first row = 0 (can't leave ring 0).

       This is an intermediate function function to find MFPT. 

    """
    
    main_diagonal = [(2*diag_m[0]) / (2*diag_m[0] + off_diag_m[0])]   
    super_diagonal = [off_diag_m[0] / (2*diag_m[0] + off_diag_m[0])]
    sub_diagonal = []
    
    #iterate, avoiding the boundaries
    for i in range(1,len(diag_m)-1):
        main_diagonal.append((2*diag_m[i]) / (2*diag_m[i] + off_diag_m[i-1] + off_diag_m[i]))
        super_diagonal.append(off_diag_m[i] / (2*diag_m[i] + off_diag_m[i-1] + off_diag_m[i]))
        sub_diagonal.append(off_diag_m[i-1] / ((2*diag_m[i] + off_diag_m[i-1] + off_diag_m[i])))
        
    n = len(diag_m) - 1
    main_diagonal.append(2*diag_m[n] / (2*diag_m[n] + off_diag_m[n-1]))
    sub_diagonal.append(off_diag_m[n-1] / (2*diag_m[n] + off_diag_m[n-1]))
        
        
    
    #These are to make the sparse matrix
    temp1 = np.array([main_diagonal, [1] + super_diagonal, sub_diagonal + [1]])   #the three diagonals; prepend/appended 1 to get positioning right
    temp2 = np.array([0, 1, -1])                                                  #main, super, and sub, diagonals
   
    B = ss.spdiags(temp1, temp2, len(main_diagonal), len(main_diagonal)).todense()
    
    #Make top row zero, since we want FIRST passage time
    B[0,0], B[0,1] = 0,0
    
    return B



def find_first_passage_probability(t, rings, B, g):
    """ Defined in paper. Intermediate function to find MFPT """
    
    temp = 0.0
    for i in range(1,len(rings.values())):         #don't include ring_0
        temp += float((len(rings.values()[i]))) / (len(g.nodes()) -1) * (B**t)[i,0]
 
    return temp



def check_current_ring(node, rings):
    for i in range(len(rings)):
        if node in rings.values()[i]:
            return rings.keys()[i]
        
        
        
def find_numerical_ring_MFPT(start_node, end_node, g, num_trials):
    """ Input: start_node, end_node in (x, y) format. g = nx.Graph()
        Output: [mean number of rings crossed, mean number of steps taken]

        Assumes graph is fully connected.
    """
        
    steps = []
    trial = 0

    while trial <= num_trials:
        current_node = start_node
        steps_per_trial = 0
            
        while current_node != end_node: 
            current_node = random.choice(g[current_node].keys())
            steps_per_trial += 1
            
        steps.append(steps_per_trial)
        trial += 1
            
    return np.mean(steps)

