# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#---------------------------------------------------- ANALYTIC MEAN FIRST PASSAGE TIME ------------------------------------------------------------------

""" Based on "Random Walks on Complex Networks" by Jae Dong Noh week ending 19 MARCH 2004
"""

import networkx as nx
import numpy as np
from scipy.linalg import eig

def __main__():
    g = nx.read_dot('shortened_winter_wheat_56mresolution_upstate_ny_20box_radius_inf.dot')
    cc = nx.connected_component_subgraphs(g)
    for sub_graph in cc:
        if len(sub_graph.nodes()) > 2:
            A = nx.adjacency_matrix(sub_graph)
            eigenvalues = eig(A)[0]
            MFPT_matrix = find_MFPT_matrix(A, sub_graph, 0.005, 1000)
        
            fout = open('Connected Component ' + str(cc.index(sub_graph)) + '.npz', 'w+')
            np.savez(fout, A, eigenvalues, MFPT_matrix)
            fout.close
                 
    return None
        
#------------------------------------------------FUCNTIONS FOR FINDING PURELY ANALYTIC MFPT ---------------------------------------------------------------------
def K(i, g_trial):
    """K_i := Sum[A_ij, i]. That is, the number of neighbours
       of node K.

       This is inefficient -- use nx.neighbors
       I'll have to find how nodes are ordered though
       that is, which node is (i,j) in adj. Matrix.
    """
    
    return len(nx.neighbors(g_trial, g_trial.nodes()[i]))


def find_stationary_probability(node_i, N, g_trial):
    """ P_i(t = inf) = K_i / Sum(K_j) 
        N := Sum[K_i, i]
    """
    return float(K(node_i, g_trial)) / N 



def find_zeroth_moment_matrix(A, g_trial, tolerance, max_iteration):
    """ finds the zeroth order moment matrix defined by
        R_ij = Sum[P_ij(t) - P_j^inf, {t,0 inf}] as per paper
        Since we have an infinte sum, the function has a tolerance.
        In my experience, 0.05 is suitable. Convergence is slow however,
        so I've included a max iteration also.

        Input: A := np.matrix(Adjacency Matrix)
    """

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
            err_check = max(prob_matrix[:,j] - find_stationary_probability(j, N, g_trial))
            if err_check > max_error:
                max_error = err_check
            
        prob_matrix = (A / np.sum(A, 0))*prob_matrix    #P(t+1) = W* P(t), W_ij = A_ij / sum(A_kj, k)
            
    return zeroth_moment_matrix
    

def find_MFPT_matrix(A, g_trial, tolerance, max_iteration):
    """ MFPT_matrix is defined by: 
        <T_ij> := N/K_j                         for i = j
        <T_ij> := N/K_j [R_jj^(0) - R_ij^(0)]   for i != j

        where N = Sum[K_i, i], K_i = neighbours of node i, 
        R_ij^(0) = zeroth order moment matrix.
        
        R_ij^(0) contains an infinite sum, so a tolerance is needed.
        Convergence can be slow so a max_iteration is included also
        See find_zeroth_order_matrix() for more details.
        
        Input: A:= np.matrix(Adjacency matrix).
    """
    
    MFPT_matrix = find_zeroth_moment_matrix(A, g_trial, tolerance, max_iteration)                                         # will *ultimately* contain MFPT
    MFPT_matrix = np.multiply( np.diag(MFPT_matrix) - MFPT_matrix, zip(np.array(float(np.sum(A)) / np.sum(A,0)[0] )[0]))  # T_ij = (R_jj - R_ij) / p_j^inf
    np.fill_diagonal(MFPT_matrix, float(np.sum(A)) / np.sum(A,0)[0] )                                                     # T_ii = 1 / p_i^inf
    return MFPT_matrix

