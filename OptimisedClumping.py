# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

                                                   ## FUNCTIONS FOR INITIALISING GRAPHS ##
import cPickle as pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import cProfile        
import pycallgraph    #for profiling
import pydot          #for profiling
import math
    

class Timer():
   def __enter__(self): self.start = time.time()
   def __exit__(self, *args): print time.time() - self.start

            
#Makes a graph from [(x1,y1), ..], as stored in XWheatResults. Each node is labeled by a string of form 'x1 y1'. So, g['x1 y1'] return the node at (x1, y1).
def makeGraph(data):
    g = nx.Graph()
    for i in xrange(len(data)):
        g.add_node(str(data[i][0]) + ' ' + str(data[i][1]))
    return g


def dist(s1, s2):
    
    """Return dist between to nodes, input: ('x1 y1', 'x2, y2'), i.e. as strings. """
        
    temp1 = np.array([int(x) for x in s1.split()])
    temp2 = np.array([int(x) for x in s2.split()])
    
    return math.hypot(temp1[0] - temp2[0], temp1[1] - temp2[1])
   
    
#Returns array of positions, [[x1, y1], ...], of nodes in a graph g.
def listPositions(g):
    pos = [key for key in g.node]                  #list of positions as strings
    for i in xrange(len(g.node)):
        pos[i] = [int(x) for x in pos[i].split()]  #list of positions as [i,j]
    return pos

#plots nodes of graph as a scatter plot.
def plotGraph(g):
    temp = np.array(listPositions(g))
    plt.scatter(temp[:,0], temp[:,1])


#Connects node_i to node_j iff dist(node_i, node_j) <= R, for all node_i in the gprah.     
def connect(g, R):            #R = radius of infection (in units of boxes = 30m)
    pos = listPositions(g)
    
    #Sum (i,0,N), (j,0,N), i<j
    for i in xrange(len(pos)):
        for j in xrange(i+1, len(pos)):
            if math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]) <= R:
                g.add_edge(str(pos[i][0]) + ' ' + str(pos[i][1]), str(pos[j][0]) + ' ' + str(pos[j][1]))   #convert [x1, y1] -> 'x1 y1'
            
                
                
#------------------------------------------------------------------------------------------------------------------------------------------------------
                                                   ## FUNCTIONS FOR COARSE GRAINING ##
#data of form [[x1, y1], [x2, y2], ...]                
def probeLengthScale(data, r_min, r_max):
    """invesitgates how qualitative behaviour (the number of connected components: CC = Connected Component, #CC = number of CC) of graph changes as a function of 
       interaction radius. Computes the number of CC for each r, denoted CC(r), in the interval (r_min, r_max),
       and if CC(r_i) != CC(r_min), stops and prints r_i. I developed this to test coarse graining. Best to use this on a subset of data (I took the first 500 from)."""
    
    g_coarse = makeGraph(data)
    connect(g_coarse, r_min)
    
    for i in xrange(r_min, r_max):
        g = makeGraph(data)
        connect(g, i)
        if nx.connected_components(g_coarse) != nx.connected_components(g):
            print 'Different # of connected components at r =  ' + str(i)
            break
            
            
#Input: nx.Graph(). Output: list of nodes (one from each CC)
def mostConnectedNodes(g):
    "Finds which node has highest degree in each CC"
    
    connected_components = nx.connected_components(g)
    return [max(g.degree(i).iteritems(), key = lambda x:x[1])[0] for i in connected_components ]


#Input: n = node, connected_components = list of connected components, (CC)
def getNeighbors(n, connected_components):
    return [i for i in connected_components if n in i]

#Input: most_connected = list of nodes with highest degree, one from each CC
def findBiggestNeighbor(n, connected_components, most_connected):
    """Biggest = node with highest degree  """
    return [most_connected[i] for i in xrange(len(connected_components)) if n in connected_components[i]]

#Input: rips reduce CC, original CC                                
def findBiggestExternalNode(n, smallCC, bigCC, most_connected_gg):
    nearNeighbors = getNeighbors(n, smallCC)  
    farNeighbors = getNeighbors(n, bigCC)
    
    if nearNeighbors != farNeighbors:
        uncommonNeighbors = list(set(farNeighbors[0]) - set(nearNeighbors[0]))  #taking [0] deals with overbracketing issues
        
        temp = []
        for i in uncommonNeighbors: 
            temp.append(findBiggestNeighbor(i, smallCC, most_connected_gg)[0])
            temp = list(set(temp))
            
        return temp
    
    else:
        return 0   #if no neighbours, return 0. Is this bad practice?
    
def connectCoarseGraph(most_connected_gg, radius_of_infection, g_coarse):
    """Connectes nodes in g_coarse if they were connected in g_original """
    
    for i in xrange(len(most_connected_gg)):
        for j in xrange(i+1, len(most_connected_gg)):
                if dist(most_connected_gg[i], most_connected_gg[j]) <= radius_of_infection:  
                #if not most_connected_gg[i] in g[most_connected_gg[j]]:                     #check if node i was unconnected to node j in graph g
                    g_coarse.add_edge(most_connected_gg[i], most_connected_gg[j])
    
    
def connectNeighborsNeigbors(most_connected_gg, connected_components_gg, connected_components_g, g_coarse, g):
    """Connects nodes in g_coarse, if they had neighbors who were connected, and weren't connected themselves --> if the sub-clusters were connected. """
    
    for i in xrange(len(connected_components_gg)):
        
        big_node = most_connected_gg[i]                                                #this is the node with highest degree, in each CC
        small_nodes = connected_components_gg[i][0:len(connected_components_gg[i])]    #this is a list of big_nodes neighbors. Strange notation, l = list[0:end], 
        small_nodes.remove(big_node)                                                   #since python treats lists by reference                                                   
    
        for j in small_nodes:
            # If any small node has a neighbour that the big node isn't connected to (unconnected Neighbor), connect the big node to this neighbour.
            unconnectedNeighbor = findBiggestExternalNode(j, connected_components_gg, connected_components_g, most_connected_gg)
            
            for k in unconnectedNeighbor:
                if not k == 0:                                            #0 means there is no neighbor, so if there IS a neighbour.
                    if not big_node in g_coarse[k]:                              # in big_node wasn't originally (in original graph) connected to
                #if not isConnected(unconnectedNeighbor[0], big_node, g):                neighbours neighbours, connect them.
                        g_coarse.add_edge(big_node, k)
                    
                    
def convexHull(points):
    """Computes the convex hull of a set of 2D points.
 
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
 
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))
 
    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points
 
    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
 
    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
 
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
 
    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]


def areClustersConnected(boundary1, boundary2, radius_of_infection):
    """ Checks if Cluster (connected_component) 1, 2, having boundary 1, 2, are within radius of infection.
        boundaryI is a list of lists: { boundary_of_connected_component_i  }      """
    
    temp = False
    
    for i in xrange(len(boundary1)):
        for j in xrange(len(boundary2)):
            if math.hypot(boundary1[i][0] - boundary2[j][0], boundary1[i][1] - boundary2[j][1]) <= radius_of_infection:
                temp = True
                break
    return temp
            
def stringsToTuples(connected_components):
    """ Change data structure """
    
    temp = []
    for l in connected_components:    
      temp.append([tuple([int(x) for x in i.split()]) for i in l])
    
    return temp
            
def findBoundaries(connected_components):
    positions = stringsToTuples(connected_components)
    return [convexHull(i) for i in positions]


def connectClusters(connected_components_gg, g_coarse, most_connected_gg, radius_of_infection):
    """Connected Clusters if they are within the radius of infection """
    
    boundaries = findBoundaries(connected_components_gg)
    
    for i in xrange(len(boundaries)):
        for j in xrange(i+1, len(boundaries)):
            if areClustersConnected(boundaries[i], boundaries[j], radius_of_infection):
                g_coarse.add_edge(most_connected_gg[i], most_connected_gg[j])
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------
                                                   ## MAIN ##
    
def main(data, radius_of_infection, rips_radius):
    
    #Make and connect graphs
    print 'Making Graphs'
    #Don't need the initial graph anymore
    #g = makeGraph(data)                                   #Original graph
    #connect(g, radius_of_infection)
    
    gg = makeGraph(data)                                  #Rips Reduce graph (coarse graining length)
    connect(gg, rips_radius)
    
    print "Finding CC's"
    connected_components_gg = nx.connected_components(gg) #Find the CC's
    most_connected_gg = mostConnectedNodes(gg)            #Find the list of 'biggest' nodes (one for each CC)
    
    #connected_components_g = nx.connected_components(g)
    
    #Coarse grained Graph
    g_coarse = nx.Graph()
    g_coarse.add_nodes_from(most_connected_gg)
    
    """ Old Way 
    print "Coarse Graining"
    #Connect coarse grained graph
    connectCoarseGraph(most_connected_gg, radius_of_infection, g_coarse)
    connectNeighborsNeigbors(most_connected_gg, connected_components_gg, connected_components_g, g_coarse, g) """
    
    #New way - compute boundary
    connectClusters(connected_components_gg, g_coarse, most_connected_gg, radius_of_infection)
    
    return g_coarse                              # return the coarse grained graph.                        
    

# <codecell>

"""
#All cells are commented out, so that .py can be imported without evaluation

data = pickle.load(open('SpringWheatPositions'))
results = main(data, 500, 5)
pickle.dump(results, open( "CoarseGrainedSpringWheat", "wb" ) )

"""

# <codecell>

"""
data1 = pickle.load(open('WinterWheatResults'))
results = main(data1, 500, 5)
pickle.dump(results, open( "CoarseGrainedWinterWheat", "wb" ) )
"""

# <codecell>

"""
#Work with smaller data set first
data = pickle.load(open('SpringWheatResults'))
sdata = data[:5000]   #2000 takes ~ 20s
"""

# <codecell>

"""
Below is for timing / profiling

with Timer():
    results = main(sdata, 350, 1)
    
pycallgraph.start_trace()
cProfile.run('main(sdata, 350, 1)')
pycallgraph.make_dot_graph('BottleNeck.png')
"""

