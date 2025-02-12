import pgmpy
import networkx as nx
from typing import Union
from BayesNet import BayesNet
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import copy
import time

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

    def maxing_out(self, cpt, var):
        if var not in list(cpt.columns):
            return ("Var not in cpt")
        varvalues = {}
        if len(cpt.index) <= 2:
            max = cpt[var].max()
            index = cpt.index[cpt[var] == max]
            group = pd.DataFrame(cpt.loc[index[0]])
            varvalues[var] = max
            group = group.transpose()
            return group, varvalues 
        if len(list(cpt.columns)) > 2:
            othervars = list(cpt.columns)[:-1] #Gets columns without p
            othervars.remove(var)
            group = cpt.groupby(othervars)['p'].agg(['idxmax', 'max']) #Groups by all other variables, gets the places where p is maximal
            values = []
            for location in group['idxmax']: #Finds the actual value of the max index
                max = cpt[var][location]
                values.append(max)
            
            varvalues[var] = values
            group = group.drop('idxmax', axis=1)

        group=group.reset_index()
        group =group.rename(columns={'max': 'p'})

        return group, pd.DataFrame(varvalues) #Returns both cpt with maxed out, and a dataframe with the maxed out variables values

    def sequence(self, selected_node, x, y, z):
        #code for sequence
        if selected_node in x:
            if list(y)[0] in y:
                if selected_node in z:
                    return True
        return False

    def fork(self, selected_node, x, y, z):
        #code for fork
        if x in self.bn.get_children(selected_node):
            if y in self.bn.get_children(selected_node):
                if selected_node in z:
                    return True
        return False

    def collider(self, graph, selected_node, x, y, z):
        #code for collider
        paths = self.get_paths(x, y)
        if selected_node in paths:
            if selected_node in paths:
                if selected_node not in z:
                    for descendent in nx.descendants(graph, selected_node):
                        if descendent in z:
                            return False
                    return True

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

    def get_paths(self, root:list, leaf:list): 
        graph = self.bn.get_interaction_graph()
        all_paths = []
        for node in root: #Get all paths from each root to each leaf
            for l in leaf:
                paths = (list(nx.algorithms.all_simple_paths(graph, node, l)))
                for path in paths:
                    all_paths.append(path) #Put all of these paths in all_paths
        for path in all_paths:
            for node in range(len(path)-1): #For every node in a path, check if next node is either child or parent of current node
                children = self.bn.get_children(path[node+1]) 
                parents = self.bn.get_parents(path[node+1]) 
                family = children + parents
                if path[node] in family: #Checks whether next node is a child or parent of this node
                    continue
                elif path[node] not in family: #If next node is not a child or parent of current node, delete the path
                    all_paths.remove(path)
                    break 
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
            next_node = min(nodes.keys(), key = lambda i: nodes[i])
            nodes.pop(next_node)
            elimination_order.append(next_node)

        #before removing it, get a list of all the neighbours from the node
        neighbours = list(graph.neighbors(next_node))

        #remove node with least amount of edges from graph
        graph.remove_node(next_node)

        #look for all neighbours and add an edge if necessary
        for pos in list(itertools.combinations(neighbours, 2)):
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
                for pos in list(itertools.combinations(neighbours, 2)):
                    if not graph.has_edge(pos[0], pos[1]):
                        dict_edges[node] += 1
            #which node has the fewest edges
            next_node = min(dict_edges.keys(), key = lambda i: dict_edges[i])
            #put it in the list of elimination order
            elimination_order.append(next_node)
            #delete this node and start again with the next node
            dict_edges.pop(next_node)

        return elimination_order

    def prune(self, query, evidence):
        self.node_prune(query,evidence)
        self.edge_prune(query, evidence)
        return self

    def node_prune(self, q, e): #Performs Node Pruning given query q and evidence e
        for node in self.bn.get_all_variables():
            if node not in q and node not in e and len(self.bn.get_parents(node)) != 0:
                self.bn.del_var(node)
        return self

    def edge_prune(self, q, e): 
        for node in e.index: #Maybe keys?
            edges = self.bn.get_children(node)
            for edge in edges: #for children of the nodes
                if edge not in q:
                    self.bn.del_edge([node, edge])
        self.prune_cpt_updater(e)
        return self

    def prune_cpt_updater(self, e):
        for node in e.index:
            cpt = self.bn.get_cpt(node)
            edges = self.bn.get_children(node)
            newcpt = cpt[e[node]==cpt[node]].reset_index(drop=True) #Gets the rows where evidence value is the same as cpt
            self.bn.update_cpt(node, newcpt) #Updates cpt to these new rows
            for edge in edges: #for children of the nodes
                cpt = self.bn.get_cpt(edge)
                newcpt = cpt[e[node]==cpt[node]].reset_index(drop=True)  #Gets the rows where evidence value is the same as cpt
                self.bn.update_cpt(edge, newcpt) #Updates cpt to these new rows
        return self

    #CPT Operations
    def factor_multiplication(self, cpt1, cpt2, allow_itself = False):

        #if set(list(cpt1.columns) + list(cpt2.columns)) == set(cpt1.columns): #Avoid multiplying with itself
        #    return cpt1
        #if set(list(cpt1.columns) + list(cpt2.columns)) == set(cpt2.columns): #Avoid multiplying with itself
        #    return cpt2
        if allow_itself == False:
            if str(cpt1.columns) == str(cpt2.columns):
                return cpt1

        variables = []
        for column in cpt1.columns: #Creates list with all variables in CPTs
            variables.append(column)
        for column in cpt2.columns:
            variables.append(column)
        variables = list(set(variables))
        variables.remove('p')
        
        tflist = [True, False]
        combi = [list(zip(variables, element))
            for element in itertools.product(tflist, repeat = len(variables))] #Get all combinations of True False and variables

        NewCpt = pd.DataFrame(columns=variables, index=range((len(combi)))) #Put all combinations of variables in dataframe
        i=0
        for element in combi: 
            for key, value in element:   
                NewCpt[key][i] = value   
            i+=1
        NewCpt['p'] = np.zeros(i)

        var1 = set(variables).intersection(cpt1)   #Gets variables that are in cpt1
        var2 = set(variables).intersection(cpt2)   #Gets variables in cpt2
            
        for j in range(i):
            checklist1 = []
            checklist2 = []
            for v1 in var1:
                check = NewCpt.loc[j, v1] #Gets value for variable in newCpt
                checklist1.append("(cpt1["+'"' + str(v1) +'"'+ ']' + '==' + str(check)+ ')') #Creates a query to find value in cpt1
            row = cpt1.loc[eval(' & '.join(checklist1))].reset_index() #Queries cpt1 to find row given values in newcpt

            try:
                p1 = (row['p'][0]) #Gets p value from row
            except: 
                p1 = np.nan

            for v2 in var2: #Same for cpt2
                check = NewCpt.loc[j, v2] 
                checklist2.append("(cpt2["+'"' + str(v2) +'"'+ ']' + '==' + str(check)+ ')')
            row = cpt2.loc[eval(' & '.join(checklist2))].reset_index() 
            try:
                p2 = (row['p'][0]) #Gets p value from row
            except: 
                p2 = np.nan

            NewCpt['p'][j] = p1*p2 #adds P1 * P2 to the newcpt
        return NewCpt.dropna(axis=0)

    def marginalization(self, cpt, var):
        cpt = cpt.drop(var, axis=1) 
        varskept = list(cpt.columns)[:-1] #Removes P from grouping
        cpt = cpt.groupby(varskept).sum() #Groups CPT by variables that are still left, then sums the p values
        return cpt.reset_index()

    def ordering(self, x, heuristic):
        if heuristic == "min-degree":
            return self.min_degree(x)
        elif heuristic == "min-fill":
            return self.min_fill(x)
        else:
            raise TypeError("Give the right heuristic, either min-degree or min-fill")
            return None

    def variable_elimination(self, q: list, heuristic = 'min-degree'):
        # All the factors relevant to the problem
        dependencies = list(set(self.get_parents(q) + q))
        dependencies = self.ordering(dependencies, heuristic)
        
        factor, final_factor = pd.DataFrame(), pd.DataFrame()
        
        for f in dependencies:
            factor = self.bn.get_cpt(f)
            # Multiply with roots
            for r in self.get_roots(f):
                if r not in q:
                    factor = self.factor_multiplication(factor, self.bn.get_cpt(r))


            # Multiply the existing result with the next factor
            if not final_factor.empty:
                final_factor = self.factor_multiplication(factor, final_factor)

                # Marginalize variables
                for p in list(final_factor.columns)[:-1]:
                    if (p not in q) and (p != f):
                        final_factor = self.marginalization(final_factor, p)
            
            if final_factor.empty:
                final_factor = factor

        for p in list(final_factor.columns)[:-1]:
            if (p not in q):
                final_factor = self.marginalization(final_factor, p)
        
        return final_factor

    def marginal_distribution(self, q: list, e):
        self.prune(q, e)
        distribution = pd.DataFrame()
        
        for var in q:
            cpt = self.variable_elimination([var])
            cpt = cpt.set_index(var)
            distribution[var] = cpt['p']/cpt['p'].sum()#evidencevalue
        distribution.index.names = ['Assignment']
        return distribution


    def mpe(self, evidence):
        self.prune(self.bn.get_all_variables(), evidence)

        q = self.bn.get_all_variables()

        for x in list(evidence.index):
            q.remove(x)

        return self.map(q, evidence)
    

    def map(self, query, evidence): 
        copynet = copy.deepcopy(self)
        cpts = {}
        querycpt = copynet.variable_elimination(query)
        varvalues = pd.DataFrame()
        for value in query:
            cpt = copynet.marginal_distribution([value], evidence)
            cpt = cpt.reset_index(drop=False)
            cpt.rename(columns = {'Assignment':value, value:'p'}, inplace = True)
            cpts[value] = cpt
        for value in query:
            querycpt = copynet.factor_multiplication(querycpt, cpts[value], allow_itself=True)
            querycpt, assignment = copynet.maxing_out(querycpt, value)
            print(varvalues)
            print(assignment)
        return querycpt


    def get_parents(self, variables:list):
        parents = []
        for var in variables:
            ancestors = (self.bn.get_parents(var))
            for parent in ancestors:
                parents.append(parent)
        for var in parents:
            ancestors = (self.bn.get_parents(var))
            for parent in ancestors:
                parents.append(parent)
        new_parents = []
        for parent in parents:
            if len(self.bn.get_parents(parent)) == 0:
                pass
            else:
                new_parents.append(parent)
        return new_parents

    def get_roots(self, variable):
        parents = self.bn.get_parents(variable)
        roots = []
        for parent in parents:
            if len(self.bn.get_parents(parent)) == 0:
                roots.append(parent) 
        return roots
       
def test():
    BNR = BNReasoner('testing/dog_problem.BIFXML')
    #test pruning
    BT = copy.deepcopy(BNR)
    BT.bn.draw_structure()
    BT.prune(query=['family-out', 'bowel-problem'], evidence=pd.Series(data={'dog-out':True}, index=['dog-out']))
    BT.bn.draw_structure()

    #test multiplication
    multiplication_cpt = BNR.factor_multiplication(BNR.bn.get_cpt('family-out'), BNR.bn.get_cpt('hear-bark'))
    print(multiplication_cpt)

    #test d-sep
    x = ["family-out"]
    y = ["bowel-problem"]
    z = ["dog-out"]

#   dsep = BNR.d_seperated(x, y, z)
#   print(dsep)

    #test indep
    x = ['bowel-problem']
    z = ['dog-out']
    y = ['family-out']
    #independent = BNR.independent(x, y, z)
    #print(independent)


    #test marg
    print(BNR.marginalization(BNR.bn.get_cpt('hear-bark'), 'dog-out'))
    #test max-out
    print(BNR.maxing_out(BNR.bn.get_cpt('hear-bark'), 'dog-out'))


    #test fac mul
    print(BNR.factor_multiplication(BNR.bn.get_cpt('family-out'), BNR.bn.get_cpt('hear-bark')))

    #test order
    #mindegree = BNR.min_degree(["Wet Grass?", "Sprinkler?", "Slippery Road?", "Rain?", "Winter?"])
    #minfill = BNR.min_fill(["Wet Grass?", "Sprinkler?", "Slippery Road?", "Rain?", "Winter?"])

    #print(mindegree)
    #print(minfill)

    #test var elim
    BT = copy.deepcopy(BNR)
    print(BT.variable_elimination(['hear-bark', 'family-out']))
    #test marg distr
    BT = copy.deepcopy(BNR)
    print(BT.marginal_distribution(['hear-bark', 'family-out'], pd.Series(data={'bowel-problem':False}, index=['bowel-problem'])))

    #test mpe

    #test MAP
    BT = copy.deepcopy(BNR)
    print(BT.map(['hear-bark', 'family-out'], pd.Series(data={'bowel-problem':False}, index=['bowel-problem']))) #hear-bark should be true, family out should be true
    return

test()

def test_varelimt1(net): #test var elimination on dog test set
    time_md = []
    for i in range(0, 10):
        start = time.time()
        net.variable_elimination(['family-out', 'hear-bark', 'light-on', 'dog-out'], heuristic = 'min-degree')
        end = time.time()
        time_md.append((end-start) * 10**3)
 
    output_md = 'Min degree mean runningtime ms: ', np.mean(time_md)
 
    time_mf = []
    for i in range(0, 10):
        start = time.time()
        net.variable_elimination(['family-out', 'hear-bark', 'light-on', 'dog-out'], heuristic = 'min-fill')
        end = time.time()
        time_mf.append((end-start) * 10**3)
 
    output_mf = "Min fill mean runningtime ms: ", np.mean(time_mf)
 
    return output_md, output_mf
 
def test_varelimt2(net): #test var elimination on our own test set
    time_md = []
    for i in range(0, 10):
        start = time.time()
        net.variable_elimination(['Family History', 'Decrease Energy', 'Stress', 'Shift Moods'], heuristic = 'min-degree')
        end = time.time()
        time_md.append((end-start) * 10**3)
 
    output_md = 'Min degree mean runningtime ms 2: ', np.mean(time_md)
 
    time_mf = []
    for i in range(0, 10):
        start = time.time()
        net.variable_elimination(['Family History', 'Decrease Energy', 'Stress', 'Shift Moods'], heuristic = 'min-fill')
        end = time.time()
        time_mf.append((end-start) * 10**3)
 
    output_mf = "Min fill mean runningtime ms 2: ", np.mean(time_mf)
 
    print(output_md, output_mf)
 
 
def test_orderingt1(net): #test ordering on dog problem test file
    startor = time.time()
    ord_deg = net.ordering(['light-on', 'bowel-problem', 'dog-out', 'hear-bark', 'family-out'], 'min-degree')
    endor = time.time()
 
    mindegree = 'The ordering given by min-degree and the time it took, respectively:', ord_deg, ((endor-startor) * 10**3)
 
    starto = time.time()
    ord_fill = net.ordering(['light-on', 'bowel-problem', 'dog-out', 'hear-bark', 'family-out'], 'min-fill')
    endo = time.time()
 
    minfill = ('The ordering given by min-fill and the time it took, respectively:', ord_fill, ((endo-starto) * 10**3))
 
    print(mindegree, minfill)
 
 
def test_orderingt2(net): #test ordering on our own test file
    startor = time.time()
    ord_deg = net.ordering(['Depression', 'Manic Depression', 'Alcohol Abuse', 'Shift Moods', 'Stress', 'Low Self Esteem', 'Family History', 'Suicide Attempt', 'Decrease Energy', 'Bad Sleep'], 'min-degree')
    endor = time.time()
 
    mindegree = 'The ordering given by min-degree and the time it took, respectively:', ord_deg, ((endor-startor) * 10**3)
 
    starto = time.time()
    ord_fill = net.ordering(['Depression', 'Manic Depression', 'Alcohol Abuse', 'Shift Moods', 'Stress', 'Low Self Esteem', 'Family History', 'Suicide Attempt', 'Decrease Energy', 'Bad Sleep'], 'min-fill')
    endo = time.time()
 
    minfill = ('The ordering given by min-fill and the time it took, respectively:', ord_fill, ((endo-starto) * 10**3))
    print(mindegree, minfill)
 
def test_marg_1(net, prune: bool):
    marg_list = []
    if prune is True:
        for i in range(0, 10): 
            start = time.time()
            net.marginal_distribution(['light-on', 'dog-out'], pd.Series(data={'hear-bark': True, 'family-out' :False}, index = ['hear-bark', 'family-out']), prune = True)
            end = time.time()
            marg_list.append((end-start) * 10**3)
    else:
        for i in range(0, 10): 
            start = time.time()
            net.marginal_distribution(['light-on', 'dog-out'], pd.Series(data={'hear-bark': True, 'family-out' :False}, index = ['hear-bark', 'family-out']), prune = False)
            end = time.time()
            marg_list.append((end-start) * 10**3)       
    return np.mean(marg_list)
 
def test_marg_2(net, prune: bool):
    marg_list = []
    if prune is True:
        for i in range(0, 10): 
            start = time.time()
            net.marginal_distribution(['Stress', 'Bad Sleep', 'Decrease Energy'], pd.Series(data={'Low Self Esteem':False, 'Manic Depression': False}, index=['Low Self Esteem', 'Manic Depression']), prune = True)
            end = time.time()
            marg_list.append((end-start) * 10**3)
    else:
        for i in range(0, 10): 
            start = time.time()
            net.marginal_distribution(['Stress', 'Bad Sleep', 'Decrease Energy'], pd.Series(data={'Low Self Esteem':False, 'Manic Depression': False}, index= ['Low Self Esteem', 'Manic Depression']), prune = False)
            end = time.time()
            marg_list.append((end-start) * 10**3)        
    return np.mean(marg_list)
 
 
def final_test_marg():
    list_marg = []
    marg1pt = test_marg_1(nettest_1, prune = True)
    list_marg.append(marg1pt)
    marg1pf = test_marg_1(nettest_1, prune = False)
    list_marg.append(marg1pf)
    marg2pt = test_marg_2(nettest_2, prune = True)
    list_marg.append(marg2pt)
    marg2pf = test_marg_2(nettest_2, prune = False)
    list_marg.append(marg2pf)
 
    print('This is the list', list_marg)
 
def final_test_var():
    lst_varelim = []
    varelim1 = test_varelimt1(nettest_1)
    lst_varelim.append(varelim1)
    varelim2 = test_varelimt2(nettest_2)
    lst_varelim.append(varelim2)
 
    print('This is the list', lst_varelim)
 
 
def final_test_ordering():
    test_orderingt1(nettest_1)
    test_orderingt2(nettest_2)
 
    return True
 
 
nettest_1 = BNReasoner('testing/dog_problem.BIFXML')
nettest_2 = BNReasoner('testing/Bay_Net_Group_26.BIFXML')
 
def cpt(which):
    cpt = nettest_2.bn.get_cpt(which)
    cpt.to_csv(f'{which}.csv')
