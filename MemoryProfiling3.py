# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

@profile
def function():
    g = nx.fast_gnp_random_graph(1000, 1./500)
    g = nx.connected_component_subgraphs(g)[0]
    A = nx.adjacency_matrix(g)
    num_nodes = A.shape[0]
    num_intervals, num_eigenvectors = 60,3
    A = A / np.sum(A, 0)    #stochastic matrix -- don't need A anymore
    eigenvalues,left_eigenvectors = eig(A, left = True, right = False)
    groups = make_groups(eigenvalues, left_eigenvectors, num_intervals , num_eigenvectors)
    
    """
    R = make_R(groups, num_nodes)
    K = make_K(groups, num_nodes, g)
    W = np.dot(R, np.dot(A, K))
    """

    A = s.csr_matrix(A)
    R = make_sparse_R(groups, num_nodes)
    K = make_sparse_K(groups, num_nodes,g)
    W = np.dot(R, np.dot(A, K))



import sys
import random
import logging
import numpy as np
from scipy.linalg import eig
import networkx as nx
import gc
from scipy import sparse as s



def coarse_grain_W(num_intervals, num_eigenvectors, g):
    A = nx.adjacency_matrix(g)
    num_nodes = A.shape[0]
    A = A / np.sum(A, 0)    #stochastic matrix -- don't need A anymore
    eigenvalues,left_eigenvectors = eig(A, left = True, right = False)
    groups = make_groups(eigenvalues, left_eigenvectors, num_intervals , num_eigenvectors)
    R = make_R(groups, num_nodes)
    K = make_K(groups, num_nodes, g)
    return np.dot(R, np.dot(A, K))


def make_sparse_R(groups, num_nodes):
    """ R_Ci := delta_C,i := 1 if node i is in group C, and 
        0 otherwise.
        
        See paper for fuller explanation.
    """
    sparse_data = []
    for group_number in xrange(len(groups)):
        for node in groups[group_number]:
            sparse_data.append([group_number, node, 1.0])
    sparse_data = np.array(sparse_data)
    return s.csr_matrix((sparse_data[:,2], (sparse_data[:,0], sparse_data[:,1])), shape=(len(groups), num_nodes))


def make_sparse_K(groups, num_nodes, g):
    sparse_data = []
    degree_distribution = nx.degree(g).values()
    for group_number in xrange(len(groups)):
        group_degree = sum([degree_distribution[member] for member in groups[group_number]])
        for node in groups[group_number]:
            sparse_data.append([node, group_number, float(degree_distribution[node]) / group_degree ])
    sparse_data = np.array(sparse_data)
    return s.csr_matrix((sparse_data[:,2], (sparse_data[:,0], sparse_data[:,1])), shape=(num_nodes, len(groups)))


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
    
    #check if vector is complex (if slightly -- im(v) < 10**-6 -- complex, set to 0)
    if (vector.imag > 1).any():                                ##  !!change this bak to 10**-6!! ##
        raise TypeError('Cannot tolerate complex eigenvalues')
    else:
        vector =  vector.astype(float)
    
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
        
        
def find_max_eigenvalues_indices(l,num_eigenvectors):
    """ Auxilary function -- to help find max eigenvectors.
        eig function returning complex --> eigenvalues not sorted.
        So, find indices of max eigenvalues, so that we can find
        corresponding eigenvectors.
        
    """
    temp = 1
    indices = []
    while temp <= num_eigenvectors + 1:
        indices.append(l.argmax() + temp - 1)
        l = np.delete(l, l.argmax())
        temp += 1
    return indices[1:]             #we don't want the biggest eigenvalue (stationary dist.)


def make_groups(eigenvalues, left_eigenvectors, num_intervals, num_eigenvectors):
    """ Makes group as defined in coarse graining procedure. Vector
        components v1,v2 are grouped together if x_n < v1,v2 < x_n+1, 
        for EACH of the first 'num_eigenvectors' eigenvectors specified.
        The subinterval (x_n, x_n+1) are defined by [max(eigenvalue) - 
        min(eigenvalue)] / num_intervals.

        Typically we take the first few non trivial eigenvectors to preserve
        the long term behaviour of the RW; i.e. take <u1|, <u2|, <u3|.
        
        Roughly speaking, groups <u^alpha_i | and <u^alpha_j | together
        if they are roughly equal to each other, for each alpha.
        
        Output: groups contain the node INDICES. E.g. groups = [[4,5,6], [7,6]]
        mean the 4th, 5th etc nodes.

    """
    
    groups = []
    indices = find_max_eigenvalues_indices(eigenvalues, num_eigenvectors)
    partitioned_vectors = [partition(left_eigenvectors[:,i], num_intervals) for i in indices]

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
    all_vector_components = set(range(len(left_eigenvectors[:,0])))
    for component in all_vector_components - flattened_groups:
        groups.append([component])
        
    return groups


        
def make_K(groups, num_nodes, g):
    """ K_iC := k_i / sum(k_j for j in group C) * delta_C,i
        where k_i is the degree of node i, and delta_C_i =
        1 if node i is in group C, and 0 otherwise.

        See paper for fuller explantion
    """
    
    K = np.zeros((num_nodes, len(groups)))
    degree_distribution = nx.degree(g).values()
    for group_number in xrange(len(groups)):
        group_degree = sum([degree_distribution[member] for member in groups[group_number]])
        for node in groups[group_number]:
            K[node, group_number] = float(degree_distribution[node]) / group_degree
    return np.matrix(K)


def make_R(groups, num_nodes):
    """ R_Ci := delta_C,i := 1 if node i is in group C, and 
        0 otherwise.
        
        See paper for fuller explanation.
    """
    R = np.zeros((len(groups), num_nodes))
    for group_number in xrange(len(groups)):
        for node in groups[group_number]:
            R[group_number, node] = 1.0
    return R      



    
if __name__ == '__main__':
        function()

# <codecell>


