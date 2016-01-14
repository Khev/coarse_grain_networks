# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Profiling find_analytical_ring_MFPT
@profile
def function():
    g = nx.fast_gnp_random_graph(5000, 0.05)
    number_nodes = len(g.nodes())
    start_node = g.nodes()[0]
    tolerance = 0.1
    max_iteration = 2
    
    #find rings
    rings = {}
    for i in g.nodes():
        length = nx.shortest_path_length(g, start_node, i)
        if length not in rings.keys():                            #If ring hasn't been created already, do so
            rings[length] = []
        rings[length].append(i) 
        
    #find diag_m
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
        
    #find off_diag_m
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
        
    del g
        
    #make B
    main_diagonal = [(2*diag_m[0]) / (2*diag_m[0] + off_diag_m[0])]   
    super_diagonal = [off_diag_m[0] / (2*diag_m[0] + off_diag_m[0])]
    sub_diagonal = []
    for i in range(1,len(diag_m)-1):
        main_diagonal.append((2*diag_m[i]) / (2*diag_m[i] + off_diag_m[i-1] + off_diag_m[i]))
        super_diagonal.append(off_diag_m[i] / (2*diag_m[i] + off_diag_m[i-1] + off_diag_m[i]))
        sub_diagonal.append(off_diag_m[i-1] / ((2*diag_m[i] + off_diag_m[i-1] + off_diag_m[i])))   
    n = len(diag_m) - 1
    main_diagonal.append(2*diag_m[n] / (2*diag_m[n] + off_diag_m[n-1]))
    sub_diagonal.append(off_diag_m[n-1] / (2*diag_m[n] + off_diag_m[n-1]))
    temp1 = np.array([main_diagonal, [1] + super_diagonal, sub_diagonal + [1]])   #the three diagonals; prepend/appended 1 to get positioning right
    temp2 = np.array([0, 1, -1])                                                  #main, super, and sub, diagonals
    B = ss.spdiags(temp1, temp2, len(main_diagonal), len(main_diagonal)).todense()
    B[0,0], B[0,1] = 0,0
    
    #find MFPT
    term = 1.0
    error = 0.0
    total = 0.0
    counter = 0
    time = 1
    while term > tolerance and counter < max_iteration:
        temp = 0.0
        for i in range(1,len(rings.values())):         #don't include ring_0
            temp += float((len(rings.values()[i]))) / (number_nodes -1) * (B**time)[i,0]
        term = time*temp
        total += term
        time += 1
        counter += 1 
    return total

import networkx as nx
import numpy as np
import scipy.sparse as ss
if __name__ == '__main__':
        function()

