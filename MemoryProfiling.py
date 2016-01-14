# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Profiling Spectral Coarse Graining
@profile
def function():
    g_trial = nx.fast_gnp_random_graph(1000, 0.1)
    A = nx.adjacency_matrix(g_trial)
    W = A / np.sum(A, 0)    #stochastic matrix
    left_eigenvectors = eig(W, left = True, right = False)[1]
    num_intervals = 20
    num_eigenvectors = 3
    groups = make_groups(left_eigenvectors, num_intervals , num_eigenvectors)
    R = make_R(groups, A)
    K = make_K(groups, A)
    return np.dot(R, np.dot(W, K))
    
    
def partition(vector, num_intervals):
    """ Partitions the elements of a vector into
        bins of length (max(vector) - min(vector))
        / num_intervals. 

        Output: list of lists of vector components 
                belonging to the same partition. 
                E.g. [[v1,v2,v3], [v4,v5], ...], where
                x1 < v1,v2,v3, < x2, i.e. they lie on the 
                first subinterval.
    """
    
    bins = np.linspace(min(vector), max(vector), num_intervals)
    keys = range(1,len(bins)+1)
    vals = [[] for i in range(len(keys))]
    partitions = dict(zip(keys, vals))
    which_bin = np.digitize(vector, bins)
    for i in range(len(which_bin)):
        partitions[which_bin[i]].append(i)
    return partitions.values()


def find_which_partition(element, partitioned_vector):
    """ Returns the partition an element is in. E.g.
        find_which_partition(v1, v) would return [v1,v2,v3]
        if vector were partitioned as described in the doc
        string for the partition function.
    """
    for partition in partitioned_vector:
        if element in partition:
            return partition
        
        
def make_groups(left_eigenvectors, num_intervals, num_eigenvectors):
    """ Makes group as defined in coarse graining procedure. Vector
        components v1,v2 are grouped together if x_n < v1,v2 < x_n+1, 
        for EACH of the first 'num_eigenvectors' eigenvectors specified.
        The subinterval (x_n, x_n+1) are defined by [max(eigenvalue) - 
        min(eigenvalue)] / num_intervals.

        Typically we take the first few non trivial eigenvectors to preserve
        the long term behaviour of the RW; i.e. take <u1|, <u2|, <u3|.
        
        Roughly speaking, groups <u^alpha_i | and <u^alpha_j | together
        if they are roughly equal to each other, for each alpha.

    """
    
    groups = []
    partitioned_vectors = [partition(left_eigenvectors[i], num_intervals) for i in range(num_eigenvectors)]  #take first NON-TRIVIAL eigenvectors

    #Add non-empty groups
    for p0_i in partitioned_vectors[0]:
        for element_p0_i in p0_i:
            sets = []
            sets.append(set(p0_i))
            for partitioned_vector in partitioned_vectors[1:]:                            #exclude first vector
                sets.append(set(find_which_partition(element_p0_i, partitioned_vector)))
            intersections = list(set.intersection(*sets))
            if len(intersections) > 1:
                if intersections not in groups:
                    groups.append(intersections)
                
    #Add elements not in groups            
    flattened_groups = set([item for sublist in groups for item in sublist])
    all_vector_components = set(range(len(left_eigenvectors[0])))
    for component in all_vector_components - flattened_groups:
        groups.append(component)
        
    return groups

#Get rid of this?
def find_similar_nodes(W, tolerance):
    groups = []
    for i in range(W.shape[0]):
        temp = [i]
        if not i in [item for sublist in groups for item in sublist]:  #flattened groups
            for j in range(i+1,W.shape[0]):
                if all(abs(W[:,i] - W[:,j]) < tolerance):
                    temp.append(j)
            groups.append(temp)
    return groups


def node_degree(node, A):
    return sum(A[:,node])


def degree_of_group(group_number, groups, A):
    if type(groups[group_number]) == list:
        return sum([node_degree(node, A) for node in groups[group_number]])
    else:
        return node_degree(groups[group_number], A)


def make_K(groups, A):
    K = np.zeros((A.shape[0], len(groups)))
    for node in range(A.shape[0]):
        for group_number in range(len(groups)):
            if type(groups[group_number]) == list:
                if node in groups[group_number]:     #if node in group
                    K[node, group_number] = float(node_degree(node, A)) / degree_of_group(group_number, groups, A)
            else:
                if node == groups[group_number]:
                    K[node, group_number] = float(node_degree(node, A)) / degree_of_group(group_number, groups, A)
    return np.matrix(K)



def make_R(groups, A):
    R = np.zeros((len(groups), A.shape[0]))
    for group_number in range(len(groups)):
        for node in range(A.shape[0]):
            if type(groups[group_number]) == list:     #group contains lists and int's, e.g. [[1,2,3], 4]
                if node in groups[group_number]:
                    R[group_number, node] = 1
            else:
                if node == groups[group_number]:
                    R[group_number, node] = 1
    return R



def coarse_grain_W(num_intervals, num_eigenvectors, A):
    W = A / np.sum(A, 0)    #stochastic matrix
    left_eigenvectors = eig(W, left = True, right = False)[1]
    groups = make_groups(left_eigenvectors, num_intervals , num_eigenvectors)
    R = make_R(groups, A)
    K = make_K(groups, A)
    return np.dot(R, np.dot(W, K))



def inspect_coarse_graining(num_intervals, num_eigenvectors, A):
    """ Prints some quantitative comparisons between the 
        coarse grained and non - coarse grained graphs.
    """
    
    W_tilde = coarse_grain_W(num_intervals, num_eigenvectors, A)
    W = A / np.sum(A, 0)
    print 'Dimension [After, Before]: ' + str([W_tilde.shape[0], W.shape[0]])
    l_tilde,l = eig(W_tilde)[0], eig(W)[0]
    print 'Difference in eigenvalues: ' + str(abs((l_tilde - l[:len(l_tilde)])[:num_eigenvectors]))
    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Eigenvalue')
    plt.title('For ' + str(num_intervals) + ' intervals')
    plt.plot(l_tilde[:num_eigenvectors+8], 'ko')
    plt.plot(l[:num_eigenvectors+8], 'ro')

     
import networkx as nx
import CoarseGrainLibrary as lib
import numpy as np
from scipy.linalg import eig
if __name__ == '__main__':
        function()

# <codecell>

"""
#Profiling find_analytic_MFPT
@profile
def function():
    g_trial = nx.fast_gnp_random_graph(10000, 0.05)
    A = nx.adjacency_matrix(g_trial)
    tolerance = 0.5
    max_iteration = 3
    N = float(np.sum(A))                                 #used in finding stationary probabilities
    max_error = 2*tolerance
    counter = 0
    row = A.shape[0]
    prob_matrix = np.matrix(np.identity(row))            #P(t = 0)_ij = Delta_ij 
    stationary_prob = np.array(np.sum(A,0)[0] / float(np.sum(A)))[0]
    zeroth_moment_matrix = np.matrix(np.vstack(tuple([-stationary_prob for i in xrange(len(stationary_prob))])).T)  # - p_j^inf
    del stationary_prob
    
    while max_error > tolerance and counter < max_iteration:
        counter += 1
        max_error = 0
        err_check = 0.0
        zeroth_moment_matrix += prob_matrix
        
        for j in xrange(row):
            err_check = max(prob_matrix[:,j] - lib.find_stationary_probability(j, N, g_trial))
            if err_check > max_error:
                max_error = err_check
            
        prob_matrix = lib.find_probability_matrix_recursively(A, g_trial, prob_matrix)
    
    MFPT_matrix = zeroth_moment_matrix                                  
    MFPT_matrix = np.multiply( np.diag(MFPT_matrix) - MFPT_matrix, zip(np.array(float(np.sum(A)) / np.sum(A,0)[0] )[0]))  # T_ij = (R_jj - R_ij) / p_j^inf
    np.fill_diagonal(MFPT_matrix, float(np.sum(A)) / np.sum(A,0)[0] )  
    return MFPT_matrix


import networkx as nx
import CoarseGrainLibrary as lib
import numpy as np
if __name__ == '__main__':
        function()
"""

