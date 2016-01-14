# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

                                                   ## FUNCTIONS FOR INITIALISING GRAPHS ##
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import time
import cProfile
import pycallgraph
import pydot
import math
    
#Timer
class Timer():
   def __enter__(self): self.start = time.time()
   def __exit__(self, *args): print time.time() - self.start

            
#Makes a graph from [(x1,y1), ..], as stored in XWheatResults. Each node is labeled by a string of form 'x1 y1'. So, g['x1 y1'] return the node at (x1, y1).
def makeGraph(data):
    g = nx.Graph()
    for i in xrange(len(data)):
        g.add_node(str(data[i][0]) + ' ' + str(data[i][1]))
    return g
   
    
#Returns array of positions, [[x1, y1], ...], of nodes in a graph g.
def listPositions(g):
    pos = [key for key in g.node]                  #list of positions as strings
    for i in xrange(len(g.node)):
        pos[i] = [int(x) for x in pos[i].split()]  #list of positions as [i,j]
    return pos


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
    
    
def connectCoarseGraph(most_connected_gg, g, g_coarse):
    """Connectes nodes in g_coarse if they were connected in g_original """
    
    for i in xrange(len(most_connected_gg)):
        for j in xrange(i+1, len(most_connected_gg)):
                if not most_connected_gg[i] in g[most_connected_gg[j]]:              #check if node i was unconnected to node j in graph g
                    g_coarse.add_edge(most_connected_gg[i], most_connected_gg[j])
    
    
def connectNeighborsNeigbors(most_connected_gg, connected_components_gg, g_coarse, g):
    """Connects nodes in g_coarse, if they had neighbors who were connected, and weren't connected themselves --> if the sub-clusters were connected. """
    
    for i in xrange(len(connected_components_gg)):
        
        big_node = most_connected_gg[i]                                                #this is the node with highest degree, in each CC
        small_nodes = connected_components_gg[i][0:len(connected_components_gg[i])]    #this is a list of big_nodes neighbors. Strange notation, l = list[0:end], 
        small_nodes.remove(big_node)                                                   #since python treats lists by reference                                                   
    
        for j in small_nodes:
            # If any small node has a neighbour that the big node isn't connected to (unconnected Neighbor), connect the big node to this neighbour.
            unconnectedNeighbor = findBiggestExternalNode(j, connected_components_gg, connected_components_g, most_connected_gg)
            
            if not unconnectedNeighbor == 0:                                           #0 means there is no neighbor, so if there IS a neighbour.
                if not big_node in g[unconnectedNeighbor[0]]:                          # in big_node wasn't originally (in original graph) connected to
                #if not isConnected(unconnectedNeighbor[0], big_node, g):                neighbours neighbours, connect them.
                    g_coarse.add_edge(big_node, unconnectedNeighbor)
                    
                    
#------------------------------------------------------------------------------------------------------------------------------------------------------
                                                   ## MAIN ##
    
def main(data, radius_of_infection, rips_radius):
    
    #Make and connect graphs
    print 'Making Graphs'
    g = makeGraph(data)                                   #Original graph
    connect(g, radius_of_infection)
    gg = makeGraph(data)                                  #Rips Reduce graph
    connect(g, rips_radius)
    
    print "Finding CC's"
    connected_components_gg = nx.connected_components(gg) #Find the CC's
    most_connected_gg = mostConnectedNodes(gg)            #Find the list of 'biggest' nodes (one for each CC)
    
    #Coarse grained Graph
    g_coarse = nx.Graph()
    g_coarse.add_nodes_from(most_connected_gg)
    
    print "Coarse Graining"
    #Connect coarse grained graph
    connectCoarseGraph(most_connected_gg, g, g_coarse)
    connectNeighborsNeigbors(most_connected_gg, connected_components_gg, g_coarse, g)
    
    return [g, gg, g_coarse]                              # return a list of the graphs                         
    

#----------------------------------------------------------------------------------------------------------------------

with Timer():
    data = pickle.load(open('SpringWheatResults'))
with Timer():
    results = main(data, 500, 1 )


