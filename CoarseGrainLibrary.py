# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

                                                   ## FUNCTIONS FOR INITIALISING GRAPHS ##
"""
    Note: Below is the full collection of algorithms I developed. There may be some inconsistencies, since
    a) as I grew more accustomed to python, my notation changed. b) As the project developed, I refined / 
    rewrote entirely / deleted, some routines to be more efficient / as they became redundant.

    I decided to lump them all into one big .py to make calling them easier. The functions are roughly listed
    in chronological order (the order in which i developled them).

    I still haven't made use of .logger

"""    
    
    
import cPickle as pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import cProfile        
#import pycallgraph    #for profiling
#import pydot          #for profiling
import math
import Image
import random
import scipy.sparse as ss


#For profiling
def profile(string):
    """ Time profiler.

        Input: string as 'f(arg1, arg2, ...)'
        Ouput: produces time stats and Bottleneck.png
    """
    
    pycallgraph.start_trace()
    cProfile.run(string)
    pycallgraph.make_dot_graph('BottleNeck.png')
    
    
    
def compare(function1, function2, *args):
    """ Returns the runtime of two functions
    """
    with Timer():
        function1(*args)
    with Timer():
        function2(*args)
    

#Dictionary of landuse types -- NOT exhaustive (I didn't include some of the more obscure data type)
uses = {'Crop Barley/Soy Beans': 254, 
'Crop Corn/Soy Beans': 241, 
'Crop Soybeans/Oats': 240, 
'Crop Soybeans / Cotton': 239,
'Crop WinterWheat/Cotton': 238,
'Crop Barely/Corn': 237,
'Crop Winter Wheat/Sorghum': 236,
'Crop Barely / Sorghum': 235,
'Crop Durum Wheat/ Sorghum': 234,
'Crop Lettuce/Barely': 233,
'Crop Lettuce/Cotton': 232,
'Crop Lettuce / Cantaloupe': 231,
'Crop Lettuce/Durum Wheat': 230,
'Pumpkins': 229,
'Lettuce': 227,
'Crop Oats/Corn': 226,
'Crop Winter Wheat / Corn': 225,
'Vetch': 224,
'Greens': 219,
'Clouds / No data': 81,
'Shrubland': 64,
'Fallow / Idle Cropland': 61,
'Sod / Grass Seed': 59,
'Herbs': 57,
'Hops': 56,
'Misc Vegetables': 47,
'Other Crops': 44,
'Buck Wheat': 39,
'Oats': 28,
'Rye': 27,
'Crop Winter Wheat / Soybeans': 26,
'Other Small Grains': 25,
'Winter Wheat': 24,
'Spring Wheat': 23,
'Durum Wheat': 22,
'Barely': 21,
'Soybeans': 5,
'Corn': 1,
'Water': 83,
'Open Water': 111
    
}



class Timer():
   def __enter__(self): self.start = time.time()
   def __exit__(self, *args): print time.time() - self.start

            
def makeGraph(data):
    
    """ Makes a graph from [(x1,y1), ..], as stored in XWheatResults.
        Each node is labeled by a string of form 'x1 y1'. So, g['x1 y1'] return the node at (x1, y1).
    """

    g = nx.Graph()
    for i in xrange(len(data)):
        g.add_node(str(data[i][0]) + ' ' + str(data[i][1]))
    return g



def dist(s1, s2):
    
    """Returns dist between to nodes, input: ('x1 y1', 'x2, y2'), i.e. as strings. """
        
    temp1 = np.array([int(x) for x in s1.split()])
    temp2 = np.array([int(x) for x in s2.split()])
    
    return math.hypot(temp1[0] - temp2[0], temp1[1] - temp2[1])

   
    
#Returns array of positions, [[x1, y1], ...], of nodes in a graph g.
def listPositions(g):
    
    """ Returns array of positions, [[x1, y1], ...], of nodes in a graph g. """

    pos = [key for key in g.node]                  #list of positions as strings
    for i in xrange(len(g.node)):
        pos[i] = [int(x) for x in pos[i].split()]  #list of positions as [i,j]
    return pos



#plots nodes of graph as a scatter plot.
def plotGraph(g):
    
    """ Plots nodes of graph as a scatter plot. """

    
    temp = np.array(listPositions(g))
    plt.scatter(temp[:,0], temp[:,1])
    
    
    
def string_to_list(string):
    return [int(x) for x in string.split()]



def subplot(node1, node2):
    """Intermediary Function for plot_graph_with_edges"""
    
    p1 = string_to_list(node1)
    p2 = string_to_list(node2)
    
    temp = zip(p1, p2)

    plt.plot(temp[0], temp[1], 'r', zorder = 1, lw = 1)
    plt.scatter(temp[0], temp[1], s=60, zorder = 2)
    
    
    
def plot_node_with_neighbors(node, g):
    for j in g[node]:
        subplot(node, j)
    
    temp = string_to_list(node)
    plt.scatter(temp[0], temp[1], c = 'r', s = 160)
    
    
    
def plot_graph_with_edges(g):
    """Plots networkz graph g as a scatter plot, with
       with edges joined in. Works best on complete grahps
       that is, ones with one fully connected component. 
    """
    
    for i in g.nodes():
        for j in g[i]:      #g[node] = neighbours of node
            subplot(i, j)
    

    
def connect(g, R):            
    
    """ #Connects node_i to node_j iff dist(node_i, node_j) <= R, for all node_i in
        the networkx graph g. """
    
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
    """Investigates how qualitative behaviour (the number of connected components: CC = Connected Component, #CC = number of CC) of graph changes as a function of 
       interaction radius. Computes the number of CC for each r, denoted CC(r), in the interval (r_min, r_max),
       and if CC(r_i) != CC(r_min), stops and prints r_i. I developed this to test coarse graining. Best to use this on a subset of data (I took the first 500 from).

       data of form [[x1, y1], [x2, y2], ...]                

    """
    
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
    
    """#Input: n = node, connected_components = list of connected components, (CC) """
    
    return [i for i in connected_components if n in i]



#Input: most_connected = list of nodes with highest degree, one from each CC
def findBiggestNeighbor(n, connected_components, most_connected):
    
    """Biggest = node with highest degree.
       Input: most_connected = list of nodes with highest degree, one from each CC  """
    
    return [most_connected[i] for i in xrange(len(connected_components)) if n in connected_components[i]]



#Input: rips reduce CC, original CC                                
def findBiggestExternalNode(n, smallCC, bigCC, most_connected_gg):
    
    """#Input: rips reduce CC, original CC """                         
    
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
                
                
                
#-----------------------------------------------------------------------------------------------------------------------------------------------
                                          ### Functions I developed to make graphs from .tif data. Far more efficient ###
                                          ### than my original way, or finding positions, and then making the graph. ###
        
def getType(x, uses):
    """Return land use type given integer, x, and dictionary of {type, integer}, uses """
    for i in uses.items():
        if x == i[1]:
            return i[0]
        
        
                
def getPositions(x, uses, data):
    
    """Return array positions, (row, column), of given data type. 

       Input is,
       x = integer corresponding to data type, uses = dictionary (see above),
       data = np.array of import from .tif. 
        
       Output is, 
       np.array of [[row, colum],...] --> do these need to be conveted to [[x1,y1],...] not sure
       how the data is input. 
        
       I should include an error message. """


    temp = []
        
    for i in xrange(len(data)):
        for j in xrange(len(data[0])):
            if data[i][j] == x:
                temp.append([i, j])   

    return np.array(temp)



def compareGraphs(g1, g2):
    
    """#Compares the quantitative properties of two graph. So I can check the coarse graining. """

    
    #Nodes and edges
    print 'Graph1: #(Nodes, Edges) = (' + str(len(g1.nodes())) + ', ' + str(len(g1.edges())) + ')'
    print 'Graph2: #(Nodes, Edges) = (' + str(len(g2.nodes())) + ', ' + str(len(g2.edges())) + ')'

    #Connected Components
    #print '\n#CCs for graph 1: ' + str(len(nx.connected_components(g1)))
    #print '#CCs for graph 2: ' + str(len(nx.connected_components(g2)))
    
    plt.hist([len(i) for i in nx.connected_components(g1)])
    plt.hist([len(i) for i in nx.connected_components(g2)])
    plt.title('Cluster Size')
    plt.xlabel('Cluster Size')
    plt.ylabel('#Cluster')
    show()
    
    #Degree Distribution
    plt.hist(nx.degree_histogram(g1))
    plt.hist(nx.degree_histogram(g2))
    plt.title('Degree Distribution' )
    plt.xlabel('Degree')
    plt.ylabel('#Nodes')
    show()
    
    #Betweeness --- this is by far the most compuationally demanding.
    plt.hist(nx.betweenness_centrality(g1, normalized = False).values())
    plt.hist(nx.betweenness_centrality(g2, normalized = False).values())
    plt.title('Distribution of Betweenness' )
    plt.xlabel('Betweenness')
    plt.ylabel('#Nodes')
    show()        
        
        
                
                
def connect_four_neighbors(data, i, j, x, g, r):
    
    """ This is an auxilliary function for make_graph_from_array defined below.
        It check the four neighbouring elements of (i,j) element in .tif data,
        and if it has the same landuse type, adds it as a node to graph g, and
        connects the two nodes. See make_graph_from_array below for a fuller
        explanation
    """
    
    try:
        if data[i+r][j+r] == x:
            if not g.has_node(str(i+r) + ' ' + str(j+r)):
                g.add_node(str(i+r) + ' ' + str(j+r))
            
            g.add_edge(str(i) + ' ' + str(j), str(i+r) + ' ' + str(j+r))
    except IndexError:
        pass
    
        
    try:
        if data[i][j+r] == x:
            if not g.has_node(str(i) + ' ' + str(j+r)):
                g.add_node(str(i) + ' ' + str(j+r))
            
            g.add_edge(str(i) + ' ' + str(j), str(i) + ' ' + str(j+r))
    except IndexError:
        pass
        
    try:  
        if data[i-r][j+r] == x and (i-r) > 0:
            if not g.has_node(str(i-r) + ' ' + str(j+r)):
                g.add_node(str(i-r) + ' ' + str(j+r))
            
            g.add_edge(str(i) + ' ' + str(j), str(i-r) + ' ' + str(j+r))
    except IndexError:
        pass
    
        
    try:
        if data[i-r][j] == x and (i-r) > 0:
            if not g.has_node(str(i-r) + ' ' + str(j)):
                g.add_node(str(i-r) + ' ' + str(j))
            
            g.add_edge(str(i) + ' ' + str(j), str(i-r) + ' ' + str(j))
    except IndexError:
        pass
    
    

    
def make_hardcut_off_graph_from_array(data, x, r):
    """Input: data = np.array import from landuse.tif
              x = landuse type (integer)
              r = coarse graining radius, in units of 'boxes'.
   
       Output: nx.Graph() with nodes representing landuse type, connected within radius

       Outline of Algorithm: check if element (i,j) is of desired type. If yes, add to graph.
                             Then find neighbours. Since running from left to right, top to
                             bottom, only need to check nodes in 'lower right' quadrant. That
                             is, elements {(i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1)} for 'inner'
                             layers and '{(i+1, j), (i, j+1)}' for 'outermost' layer. So, 'outermost'
                             layer doesn't contain elements touching the middle element 'diagonally',
                             that is, at sqrt(2)R from middle element.

       Comments: I've used 'try: and except: ' liberally to deal with the edges. This seemed most
                 sensible, since the 'edge' is variable (since I'm connecting neighbors at a variable r)
                 Maybe generators are better for this?

    """
    
    
    g = nx.Graph()
    
    for i in xrange(len(data)):
        for j in xrange(len(data[0])):
        
            if data[i][j] == x:
                if not g.has_node(str(i) + ' ' + str(j)):
                    g.add_node(str(i) + ' ' + str(j))
            
    
        #Outer Layer has only 2 neighbors
                try:
                    if data[i+r][j] == x:
                        if not g.has_node(str(i+r) + ' ' + str(j)):     #if node isn't already in graph, add it
                            g.add_node(str(i+r) + ' ' + str(j))         #connect them.
                
                        g.add_edge(str(i) + ' ' + str(j), str(i+r) + ' ' + str(j))
                except IndexError:
                    pass

                try:
                    if data[i][j+r] == x:
                        if not g.has_node(str(i) + ' ' + str(j+r)):
                            g.add_node(str(i) + ' ' + str(j+r))
                
                        g.add_edge(str(i) + ' ' + str(j), str(i) + ' ' + str(j+r))
                except IndexError:
                    pass
        
        
        #Inner Layers have 4 neighbours.
                for n in range(1,r):
                    connect_four_neighbors(data, i, j, x, g, n)
        
    return g 



def make_graph_with_tail_from_array(data, x, r):
    """ This is the same of make_hard_cut_off_graph_from_array_, except the connectivity r
        obeys an exponential decay, with a characteristic distance r_characteristic. This 
        characteristic distance is taken to be  half of the radius given.

    """
    
    g = nx.Graph()
    r_characteristic = r/2
    
    for i in xrange(len(data)):
        for j in xrange(len(data[0])):
        
            if data[i][j] == x:
                if not g.has_node(str(i) + ' ' + str(j)):
                    g.add_node(str(i) + ' ' + str(j))
                    
                    r_eff = np.random.exponential(r_characteristic)
            
    
        #Outer Layer has only 2 neighbors
                try:
                    if data[i+r_eff][j] == x:
                        if not g.has_node(str(i+r_eff) + ' ' + str(j)):     #if node isn't already in graph, add it
                            g.add_node(str(i+r_eff) + ' ' + str(j))         #connect them.
                
                        g.add_edge(str(i) + ' ' + str(j), str(i+r_eff) + ' ' + str(j))
                except IndexError:
                    pass

                try:
                    if data[i][j+r_eff] == x:
                        if not g.has_node(str(i) + ' ' + str(j+r_eff)):
                            g.add_node(str(i) + ' ' + str(j+r_eff))
                
                        g.add_edge(str(i) + ' ' + str(j), str(i) + ' ' + str(j+r_eff))
                except IndexError:
                    pass
        
        
        #Inner Layers have 4 neighbours.
                for n in range(1,int(math.ceil(r_eff))):
                    connect_four_neighbors(data, i, j, x, g, n)
        
    return g 



def probe_length_scale(data, data_type, r_min, r_max, dr):
    """ 
        Investigates how the number of connected components changes as a
        function of interaction radius. 

        Input:  data = original data, in np.array form, as imported from .tif
                data_type = lanuse type (integer, e.g. winter wheat = 24)
                r_min = starting radius of connectivity
                r_max = ending radius of connectivity
                dr = radius increment
        
        Output: list of [[radius, #CC's, size of biggest CC],...]
    """
    
    cluster_data = []                                              #I use the terms 'cluster' and 'connected component interchangeably'
    
    for r in range(r_min, r_max + 1, dr):                          #r_max + 1 since python does, range(1,2) = 1, range(1,3) = (1,2), 
        g_r = fn.make_graph_from_array(data, data_type, r)
        CCs = nx.connected_components(g_r)
        num_CCs = len(CCs)
        size_biggest_CC = len(CCs[0])                              #we know that CC's are arrangest in order of decreasing magnitude,
        cluster_data.append([r, num_CCs, size_biggest_CC])         #so the first element is the biggest cluster
        
    return cluster_data

                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------
                                                   ## MAIN ##
    
def coarse_grain_graph(data, data_type, radius_of_infection, rips_radius):
    
    """This is my 'first attempt' main function. As I refined the algorithm, it changed,
       but I left the original form as comments, for comparison. I'll have to clean it 
       up properly soon.

       Input: data = np.array(.tif data), rips_radius = coarse graining length
        
       Output: the coarse grained networkx graph

    """
    
    #Make and connect graphs
    print 'Making Graphs'                                   
    
    #g = makeGraph(data)                                   #Original graph
    #connect(g, radius_of_infection)
    
    #gg = makeGraph(data)                                  #Rips Reduce graph (coarse graining length)
    #connect(gg, rips_radius)

    gg = make_graph_from_array(data, data_type, rips_radius)
    
    print "Finding CC's"
    connected_components_gg = nx.connected_components(gg) #Find the CC's
    most_connected_gg = mostConnectedNodes(gg)            #Find the list of 'biggest' nodes (one for each CC)
    
    #connected_components_g = nx.connected_components(g)
    
    #Coarse grained Graph
    g_coarse = nx.Graph()
    g_coarse.add_nodes_from(most_connected_gg)
    
    """ Old Way .
    print "Coarse Graining"
    #Connect coarse grained graph
    connectCoarseGraph(most_connected_gg, radius_of_infection, g_coarse)
    connectNeighborsNeigbors(most_connected_gg, connected_components_gg, connected_components_g, g_coarse, g) 
    """
    
    #New way - compute boundary
    connectClusters(connected_components_gg, g_coarse, most_connected_gg, radius_of_infection)
    
    return g_coarse                              # return the coarse grained graph.



#----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                 ### Random Walk Functions ###
    
def random_walk(g, num_steps):
    
    """ Perform Random Walk on graph g. Return distance between start and end node """
    
    start = random.choice(g.nodes())
    
    #Avoid starting in a node with no edges
    while len(g[start].keys()) == 0:
        start = random.choice(g.nodes())
  
    current_position = start
    
    for i in xrange(num_steps):
        current_position = random.choice(g[current_position].keys())

    end  = current_position

    return  dist(start, end)



def find_super_node(node, g_r, connected_components_g_r):
    
    """ Returns the supernode (which is the node with highest degree, in a given cluster)
        of a given node.

        Input: g_r = networkx graph connected at coarsening length r
               
    """
    
    most_connected = [max(g_r.degree(i).iteritems(), key = lambda x:x[1])[0] for i in connected_components_g_r ]
    
    for i in xrange(len(connected_components_g_r)):
        if node in connected_components_g_r[i]:
            return most_connected[i]
    
    

def random_walk_non_coarse_grained(g_R, g_r, num_steps, connected_components):
    """ A minor variant on random walk. Changed to measure distance
        between nearest supernode of starting and end nodes -- since
        the coarse grained graph represents all such nodes by their supernodes.
    """
    
        
    start = random.choice(g_R.nodes())
    
    #Avoid starting in a node with no edges
    while len(g_R[start].keys()) == 0:
        start = random.choice(g_R.nodes())
  
    current_position = start
    
    for i in xrange(num_steps):
        current_position = random.choice(g_R[current_position].keys())

    end  = current_position
    return  dist(find_super_node(start, g_r, connected_components), find_super_node(end, g_r, connected_components))
    

# <codecell>

def connect_four_neighbors_tuple(data, i, j, x, g, r):
    
    """ This is an auxilliary function for make_graph_from_array defined below.
        It check the four neighbouring elements of (i,j) element in .tif data,
        and if it has the same landuse type, adds it as a node to graph g, and
        connects the two nodes. See make_graph_from_array below for a fuller
        explanation
    """
    
    try:
        if data[i+r][j+r] == x:
            if not g.has_node((i+r, j+r)):
                g.add_node((i+r, j+r))
            
            g.add_edge((i, j), (i+r, j+r))
    except IndexError:
        pass
    
        
    try:
        if data[i][j+r] == x:
            if not g.has_node((i, j+r)):
                g.add_node((i, j+r))
            
            g.add_edge((i, j), (i, j+r))
    except IndexError:
        pass
        
    try:  
        if data[i-r][j+r] == x and (i-r) > 0:
            if not g.has_node((i-r, j+r)):
                g.add_node((i-r, j+r))
            
            g.add_edge((i, j), (i-r, j-r))
    except IndexError:
        pass
    
        
    try:
        if data[i-r][j] == x and (i-r) > 0:
            if not g.has_node((i-r, j)):
                g.add_node((i-r, j))
            
            g.add_edge((i, j), (i-r, j))
    except IndexError:
        pass
    
    

    
def make_hardcut_off_graph_from_array_tuple(data, x, r):
    """Input: data = np.array import from landuse.tif
              x = landuse type (integer)
              r = coarse graining radius, in units of 'boxes'.
   
       Output: nx.Graph() with nodes representing landuse type, connected within radius

       Outline of Algorithm: check if element (i,j) is of desired type. If yes, add to graph.
                             Then find neighbours. Since running from left to right, top to
                             bottom, only need to check nodes in 'lower right' quadrant. That
                             is, elements {(i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1)} for 'inner'
                             layers and '{(i+1, j), (i, j+1)}' for 'outermost' layer. So, 'outermost'
                             layer doesn't contain elements touching the middle element 'diagonally',
                             that is, at sqrt(2)R from middle element.

       Comments: I've used 'try: and except: ' liberally to deal with the edges. This seemed most
                 sensible, since the 'edge' is variable (since I'm connecting neighbors at a variable r)
                 Maybe generators are better for this?

    """
    
    
    g = nx.Graph()
    
    for i in xrange(len(data)):
        for j in xrange(len(data[0])):
        
            if data[i][j] == x:
                if not g.has_node((i, j)):
                    g.add_node((i, j))
            
    
        #Outer Layer has only 2 neighbors
                try:
                    if data[i+r][j] == x:
                        if not g.has_node((i+r, j)):     #if node isn't already in graph, add it
                            g.add_node((i+r, j))         #connect them.
                
                        g.add_edge((i,j), (i+r, j))
                except IndexError:
                    pass

                try:
                    if data[i][j+r] == x:
                        if not g.has_node((i, j+r)):
                            g.add_node((i, j+r))
                
                        g.add_edge((i,j), (i, j+r))
                except IndexError:
                    pass
        
        
        #Inner Layers have 4 neighbours.
                for n in range(1,r):
                    connect_four_neighbors_tuple(data, i, j, x, g, n)
        
    return g 

# <codecell>

#------------------------------------------------FUCNTIONS FOR FINDING PURELY ANALYTIC MFPT ---------------------------------------------------------------------
def find_numerical_MFPT(start_node, end_node, g, num_of_trials):
    """Input: g = nx.Graph()  
    """
    
    FPT = []                                            #will contain FPT of each trial
    for trial in xrange(num_of_trials):
        current_node = start_node
        FPT_i = 0                                       #FPT per trial
        while current_node != end_node:
            current_node = random.choice(g[current_node].keys())    #pick a neighbour at random
            FPT_i += 1
        FPT.append(FPT_i)
        
    return np.mean(FPT)


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
            
        prob_matrix = (A / np.sum(A, 0))*prob_matrix    
            
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
    
    MFPT_matrix = find_zeroth_moment_matrix(A, g_trial, tolerance, max_iteration)                                   # will *ultimately* contain MFPT
    MFPT_matrix = np.multiply( np.diag(MFPT_matrix) - MFPT_matrix, zip(np.array(float(np.sum(A)) / np.sum(A,0)[0] )[0]))  # T_ij = (R_jj - R_ij) / p_j^inf
    np.fill_diagonal(MFPT_matrix, float(np.sum(A)) / np.sum(A,0)[0] )                                               # T_ii = 1 / p_i^inf
    return MFPT_matrix

# <codecell>

#---------------------------------------------------- COARSE GRAINING WITH RING STRUCTURE ------------------------------------------------------------------
""" Based on "Ring structures and mean first passage time in networks"
    by Andrea Baronchelli and Vittorio Loreto
"""


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
    number_nodes = len(g.nodes())              #can delete g hereafter
    
    #del diag_m      # hardly makes a difference?
    #del off_diag_m  
    
    term = 1.0
    error = 0.0
    total = 0.0
    counter = 0
    time = 1
    
    while term > tolerance and counter < max_iteration:
        term = time*find_first_passage_probability(time, rings, B, number_nodes)
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



def find_first_passage_probability(t, rings, B, number_nodes):
    """ Defined in paper. Intermediate function to find MFPT """
    
    temp = 0.0
    for i in range(1,len(rings.values())):         #don't include ring_0
        temp += float((len(rings.values()[i]))) / (number_nodes -1) * (B**t)[i,0]
 
    return temp



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

# <codecell>


