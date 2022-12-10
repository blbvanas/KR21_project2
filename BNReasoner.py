from typing import Union
from BayesNet import BayesNet
import networkx as nx
from itertools import combinations

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net
    # TODO: This is where your methods should go

    def sequence(self, path):
        #code for sequence
        _x, _y, _z = path
        A = _y in self.bn.get_children(_x)
        B = _z in self.bn.get_children(_z)
        return A and B 

    def fork(self, path):
        #code for fork
        _x, _y, _z = path
        AB =  self.bn.get_children(_y) == (_x, _z)
        return AB

    def collider(self, path):
        #code for collider
        _x, _y, _z = path
        A = _y in self.bn.get_children(_x)
        B = _y in self.bn.get_children(_z)
        return A and B

    def path_is_closed(self, path, z):
        #check if the before functions hold, if so, path is closed
        if self.sequence(path):
            if z == path[1]:
                return True
            else:
                return False
        elif self.fork(path):
            if z == path[1]:
                return True
            else:
                return False
        elif self.collider(path):
            if z not in path:
                return True
            else:
                return False

    def d_seperated(self, x, y, evidence):
        #apply path_is_closed function on the input variables and see whether x is seperated from y given evidence. 
        graph = self.bn.get_interaction_graph()
        all_paths = list(nx.algorithms.all_simple_paths(graph, x, y))
        for path in all_paths:
            if not self.path_is_closed(path, evidence):
                print ("{x} is not d-seperated from {y} given {evidence}")
                return False
        print ("{x} is d-seperated from {y} given {evidence}")
        return True

    def independent(self, x, y, z):
        if self.d_seperated(x, y, z):
            print("{x} is independent from {y} given {z}")
            return True
        else:
            print("{x} is dependent from {y} given {z}")
            return False

    def min_degree(self, x: list):
        graph = self.bn.get_interaction_graph()
        nodes = {}
        elimination_order = []

        #how many neighbours do the variables have
        for i in x:
            nodes[i] = len(list(graph.neighbors(i)))
        
        #which node has the least amount of edges and put it in elimination order
        for j in range(len(x)):
            next_node = min(nodes.keys(), key = lambda i: nodes[i])
            nodes.pop(next_node)
            elimination_order.append(next_node)

        #before removing it, get a list of all the neighbours from the node
        neighbours = list(graph.neighbors(next_node))

        #remove node with least amount of edges from graph
        graph.remove_node(next_node)

        #look for all neighbours and add an edge if necessary
        for pos in list(combinations(neighbours, 2)):
            if not graph.has_edge(pos[0], pos[1]):
                graph.add_edge(pos[0], pos[1])
        
        return elimination_order


    def min_fill(self, x: list):
        graph = self.bn.get_interaction_graph()
        elimination_order = []
        
        #create dictionary
        dict_edges = {}
        for node in x:
            dict_edges[node] = 0

        #iterate through the variables 
        for i in range(len(x)):
            for node in dict_edges.keys():
                neighbours = list(graph.neighbors(node))
                #add number of edges to dictionary
                for pos in list(combinations(neighbours, 2)):
                    if not graph.has_edge(pos[0], pos[1]):
                        dict_edges[node] += 1
            #which node has the fewest edges
            next_node = min(dict_edges.keys(), key = lambda i: dict_edges[i])
            #put it in the list of elimination order
            elimination_order.append(next_node)
            #delete this node and start again with the next node
            dict_edges.pop(next_node)

        return elimination_order

#Pruning
def prune(net, q, e):
    node_prune(net, q, e)
    edge_prune(net, q, e)
    return net

def edge_prune(net, q, e): #TODO Update Factors see Bayes 3 slides page 28
    for node in e:
        edges = net.get_children(node)
        for edge in edges:
            net.del_edge([node, edge])
    return net

def node_prune(net, q, e): #Performs Node Pruning given query q and evidence e
    for node in BayesNet.get_all_variables(net):
        if node not in q and node not in e:
            net.del_var(node)
    return net


#def marginalization(net, variables):
#    cpt = net.get_all_cpts()
#    print(cpt)
#    for variable in factor:

    #totalp = sum(cpt["p"]) 

    
    #for variable in distribution:
    #    if variable != target_node:
    #       cpt = net.get_cpt(variable) 
    #       totalp = sum(cpt['p'])


#marginalization(net.bn, [], [])