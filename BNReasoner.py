import pgmpy
import networkx as nx
from typing import Union
from BayesNet import BayesNet
import pandas as pd
import matplotlib.pyplot as plt


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

def d_seperated(model, x, y, z):
    if nx.d_separated(model, x, y, z):
        print("X is d-seperated of Y given Z")  #overbodig, maar meer om voor ons duidelijk te hebben als we straks een eigen bn maken
        return True
    else:
        print("X is not d-seperated of Y given Z") # same here
        return False

def independent(model, x, y, z):
    if d_seperated(model, x, y, z) is True:
        print("X is independent of Y given Z") #same here
        return True
    else:
        print("X is dependent of Y given Z") #same here
        return False

def min_degree_heuristic(graph):
    nx.approximation.treewidth_min_degree(graph)
    return graph

def min_fill_heur(graph):
    print(nx.approximation.treewidth_min_fill_in(graph))
    return graph

#Pruning
def prune(net, q, e):
    node_prune(net, q, e)
    edge_prune(net, q, e)
    return net

def edge_prune(net, e): #TODO Update Factors see Bayes 3 slides page 28
    for node in e:
        edges = net.get_children(node)
        for edge in edges:
            net.del_edge([node, edge])
            #TODO : Update CPT
    return net

def node_prune(net, q, e): #Performs Node Pruning given query q and evidence e
    for node in BayesNet.get_all_variables(net):
        if node not in q and node not in e:
            net.del_var(node)
    return net

def factor_multiplication(cpt1, cpt2): #Bart
    #if cpt1 == cpt2:
    #    raise Exception('Both factors are the same') 
    cpt2 = cpt2.rename(columns={'p':'p2'})
    cpt = pd.merge(cpt1,cpt2)
    cpt['p'] = cpt['p'] * cpt['p2'] 
    cpt = cpt.drop('p2', axis=1)
    return cpt


def maginalization(cpt, var):
    cpt = cpt.drop(var, axis=1) 
    varskept = list(cpt.columns)[:-1] #Removes P from grouping
    cpt = cpt.groupby(varskept).sum() #Groups CPT by variables that are still left, 
    return cpt.reset_index()

def maxing_out(): #Bart
    return True

net = BNReasoner("C:/Users/Bart/Documents/GitHub/KR21_project2/testing/dog_problem.BIFXML")
cpt1 = net.bn.get_cpt('light-on')
cpt2 = net.bn.get_cpt('family-out')

print(factor_multiplication(cpt1, cpt1))