import pgmpy
import networkx as nx
from typing import Union
from BayesNet import BayesNet
import pandas as pd
import matplotlib.pyplot as plt
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

    def maxing_out(self, cpt_var, var): #Bart
        cpt = self.bn.get_cpt(cpt_var)
        if var not in list(cpt.columns):
            return ("Var not in cpt")
        othervars = list(cpt.columns)[:-1] #Gets columns without p
        othervars.remove(var)
        group = cpt.groupby(othervars)['p'].agg(['idxmax', 'max'])
        
        values = []
        for location in group['idxmax']:
            max = cpt[var][location]
            values.append(max)
        varvalues = pd.DataFrame()
        varvalues[var] = values
        group = group.drop('idxmax', axis=1)
        return group.reset_index(), varvalues


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

    def edge_prune(self, e): 
        for node in e:
            edges = self.bn.get_children(node)
            for edge in edges:
                self.bn.del_edge([node, edge])
                cpt = self.bn.get_cpt(edge)
                newcpt = self.bn.reduce_factor(e, cpt)
                newcpt = self.bn.update_cpt(edge, newcpt)
        return self

    def node_prune(self, q, e): #Performs Node Pruning given query q and evidence e
        for node in self.bn.get_all_variables():
            if node not in q and node not in e:
                self.bn.del_var(node)
        return self


    #Pruning
    def prune(self, q, e):
        self.edge_prune(e)
        self.node_prune(q, e)
        return net



    def factor_multiplication(self, cpt_var1, cpt_var2): #TODO: Check whether CPT's are the same
        cpt1 = self.bn.get_cpt(cpt_var1)
        cpt2 = self.bn.get_cpt(cpt_var2)
        
        cpt2 = cpt2.rename(columns={'p':'p2'})
        cpt = pd.merge(cpt1,cpt2)
        cpt['p'] = cpt['p'] * cpt['p2'] 
        cpt = cpt.drop('p2', axis=1)
        return cpt
    

    def marginalization(self, cpt_var, var):
        cpt = self.bn.get_cpt(cpt_var)
        cpt = cpt.drop(var, axis=1) 
        varskept = list(cpt.columns)[:-1] #Removes P from grouping
        cpt = cpt.groupby(varskept).sum() #Groups CPT by variables that are still left, then sums the p values
        return cpt.reset_index()


    def variable_elimination(self, q: list):
        #x = ordering()
        x = self.prune(net, q, [])
        factor, prev_factor = {}, {}
        query = set(self.bn.get_all_variables()).symmetric_difference(set(q))
        for v in query:
            factor = self.bn.get_cpt(v)
            to_sum_out = set(factor.keys()).difference(set([v]))
            if len(to_sum_out) > 1:
                for s in to_sum_out:
                    if s != 'p':
                        factor = self.marginalization(factor, s)
            if len(prev_factor): 
                factor = self.factor_multiplication(factor, prev_factor)
            prev_factor = factor
        return factor

net = BNReasoner("C:/Users/Bart/Documents/GitHub/KR21_project2/testing/dog_problem.BIFXML")
maxes, blub = net.maxing_out('dog-out', 'bowel-problem')

