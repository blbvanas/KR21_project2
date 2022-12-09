from typing import Union
from BayesNet import BayesNet
import networkx as nx


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
        _x, _y, _z = path
        A = _y in self.bn.get_children(_x)
        B = _z in self.bn.get_children(_z)
        return A and B 

    def fork(self, path):
        _x, _y, _z = path
        AB =  self.bn.get_children(_y) == (_x, _z)
        return AB

    def collider(self, path):
        _x, _y, _z = path
        A = _y in self.bn.get_children(_x)
        B = _y in self.bn.get_children(_z)
        return A and B

    def path_is_closed(self, path, z):
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
        graph = self.bn.get_interaction_graph()
        all_paths = list(nx.algorithms.all_simple_paths(graph, x, y))
        for path in all_paths:
            if not self.path_is_closed(path, evidence):
                print ("{x} is not d-seperated from {y} given {evidence}")
                return False
        print ("{x} is d-seperated from {y} given {evidence}")
        return True

    def independent(self, x, y, z):
        return self.d_seperated(x, y, z)



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



def min_degree_heuristic(graph):
    nx.approximation.treewidth_min_degree(graph)
    return graph

def min_fill_heur(graph):
    print(nx.approximation.treewidth_min_fill_in(graph))
    return graph


net = BNReasoner("C:/Users/Ellis/Documents/VU/Knowledge Representation/KR21_project2-main/testing/dog_problem.BIFXML")
#marginalization(net.bn, [], [])