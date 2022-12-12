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

    def get_paths(self, root, leaf):
        graph = self.bn.structure.copy()
        roots = []
        leaves = []
        for node in graph.nodes :
            if graph.in_degree(node) == 0: 
                roots.append(node)
            elif graph.out_degree(node) == 0:
                leaves.append(node)
        for root in roots:
            for leaf in leaves:
                all_paths = list(nx.algorithms.all_simple_paths(graph, root, leaf))
        return all_paths

    def d_seperated(self, x, y, evidence):
        #apply path_is_closed function on the input variables and see whether x is seperated from y given evidence. 
        for path in self.get_paths(x, y):
            if not self.path_is_closed(path, evidence):
                print (f"{x} is not d-seperated from {y} given {evidence}")
                return False
        print (f"{x} is d-seperated from {y} given {evidence}")
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
            next_node = min(nodes.keys(), key = lambda j: nodes[j])
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

    def ordering(self, heuristic):
        if heuristic == "min-degree":
            self.min_degree
        elif heuristic == "min-fill":
            self.min_fill
        else:
            raise TypeError("Give the right heuristic, either min-degree or min-fill")

    def prune(self, q, e):
        self.node_prune(q, e)
        self.edge_prune(q, e)
        return self
    
    def mpe(self, evidence):
        #prune edges
        pruned_net  = self.prune(self.bn.get_all_variables(), evidence)

        #get elimination order
        elimination_order = self.min_fill(pruned_net)

        #get all cpts from pruned network
        cpts = pruned_net.get_all_cpts(pruned_net)

        for variable in elimination_order:
            factor = [key for key, cpt in cpts.items() if variable in cpt.columns]
            factors_cpt = [cpts[key] for key in factor]

            factors_mult = self.multiply_factors(factors_cpt)
            
            factors_max = self.max_out()


        pass





        # query = self.bn.get_all_variables()

        # var_pruned = self.prune(query, evidence)

        # cpts = var_pruned.get_all_cpts()

        # order = ordering("min-fill")
        # for var in order:
        #     fac = []
        #     delete = []
        #     # get factors which contain variable
        #     for key, value in var_pruned.items():
        #         if var in value.columns:
        #             fac.append(value)
        #             delete.append(key)

        #     factor, rowsmult = self.multiply_factors(fac)
        #     rows_multiplied += rowsmult

        #     for variables in delete:
        #         del cpts[variables]

            


#Pruning


# def edge_prune(net, q, e): #TODO Update Factors see Bayes 3 slides page 28
#     for node in e:
#         edges = net.get_children(node)
#         for edge in edges:
#             net.del_edge([node, edge])
#     return net

# def node_prune(net, q, e): #Performs Node Pruning given query q and evidence e
#     for node in BayesNet.get_all_variables(net):
#         if node not in q and node not in e:
#             net.del_var(node)
#     return net


#def marginalization(net, variables):
#    cpt = net.get_all_cpts()
#    print(cpt)
#    for variable in factor:

    #totalp = sum(cpt["p"]) 

    
    #for variable in distribution:
    #    if variable != target_node:
    #       cpt = net.get_cpt(variable) 
    #       totalp = sum(cpt['p'])



def test_function(filename, var1, var2, Q, e):
    BNR = BNReasoner(filename)
    TestBN = BNR.bn

    #test pruning


    #test d-sep
    # x = ["Winter?"]
    # y = ["Wet Grass?"]
    # z = ["Winter?"]

    # dsep = BNR.d_seperated(x, y, z)
    # print(dsep)

    # x = ['bowel-problem']
    # z = ['dog-out']
    # y = ['family-out']
    # d_separated = BNR.d_seperated(x, y, z)
    # print(d_separated)

    #test indep
    # x = ['bowel-problem']
    # z = ['dog-out']
    # y = ['family-out']
    # independent = BNR.independent(x, y, z)

    #test marg

    #test max-out

    #test fac mul

    #test order
    # mindegree = BNR.min_degree(["Wet Grass?", "Sprinkler?", "Slippery Road?", "Rain?", "Winter?"])
    # minfill = BNR.min_fill(["Wet Grass?", "Sprinkler?", "Slippery Road?", "Rain?", "Winter?"])

    # print(mindegree)
    # print(minfill)

    #test var elim

    #test marg distr

    #test mpe

    #test map
filename_dog = 'testing/dog_problem.BIFXML'
filename_lec1 = 'testing/lecture_example.BIFXML'

BN_dog = test_function(filename = filename_lec1, var1 = 'dog-out', var2 = 'family-out', Q = [], e = {})

