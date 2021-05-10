# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:35:08 2019
Updated on 10 May 2021
@author: jrhuggenaa
@author: yzhang

"""

import cplex
from cplex.exceptions import CplexError

import sys
import csv

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import postprocess_MIP_sol_AMAI as CODE_POST_PROCESS

# The data, DATA_TABLE[-1] contains the target, others are features
FEATURE_NAMES = []
DATA_TABLE = []
TARGETS = []
OPTIMIZE_TABLE = []

def read_file(file_name):
    global DATA_TABLE, FEATURE_NAMES, TARGETS

    data = []
    header = True
    with open(file_name, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if header is True:
                FEATURE_NAMES = [str(i) for i in row]
            if header is False:
                #print [i for i in row]
                row = [float(i) for i in row]
                if row[-1] not in TARGETS:
                    TARGETS.append(row[-1])
            header = False
            data.append(row)

    DATA_TABLE = data

def read_optimize_file(file_name):
    global OPTIMIZE_TABLE

    data = []
    header = True
    with open(file_name, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if header is False:
                row = [float(i) for i in row]
            header = False
            data.append(row)

    OPTIMIZE_TABLE = data

def learn_tree(depth):
    global TARGETS
    dat = np.array(DATA_TABLE)
    x = pd.DataFrame(data=dat[1:,0:-1])
    y = pd.DataFrame(data=dat[1:,-1])
    
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(x, y)
    
    TARGETS = tree.classes_

    return tree

def auction_size():
    return len(OPTIMIZE_TABLE) - 1

def get_feature_value(row, feature):
    for i in range(len(FEATURE_NAMES)):
        if FEATURE_NAMES[i] == feature:
            return OPTIMIZE_TABLE[row+1][i]
    return 0.0;

def featuremap():
    features = []

    features.append("LotNrRel")
    features.append("Allocate")
    features.append("EstValue")
    features.append("StartPrice")
    features.append("LotsSale")
    features.append("LotsSaleMain")
    features.append("LotsSaleSub")
    features.append("Weekday")
    features.append("SP.EV")
#    features.append("Rel_EV")
    return features

FEATURE_MAP = featuremap()

def get_feature(tree, index):
    return FEATURE_MAP[tree.tree_.feature[index]]

def calc_feature(tree, row, node):
    col_names = []
    col_values = []

    num_rows = auction_size()
    
    feature = get_feature(tree, node)
    if "LotNrRel" == feature:
        for index in range(num_rows):
            col_names.extend(["index_" + str(row) + "_" + str(index)])
            col_values.extend([float(index) / float(num_rows)])
        return col_names, col_values
    if "StartPrice" == feature:
        col_names.extend(["startprice_" + str(row)])
        col_values.extend([1.0])
        return col_names, col_values
    if "SP.EV" == feature:
        col_names.extend(["startprice_" + str(row)])
        col_values.extend([1.0 / float(get_feature_value(row, "EstValue"))])
        return col_names, col_values
    
    return [],[get_feature_value(row, feature)]

def get_leaf_prediction(tree, index):
    max_value = -1
    max_index = -1
    values = tree.tree_.value[index]
    for i in range(len(values[0])):
        if values[0][i] > max_value:
            max_value = values[0][i]
            max_index = i
    #print index, values, max_index, TARGETS[max_index]
    return TARGETS[max_index]

def get_node_constant(tree, index):
    return float(tree.tree_.threshold[index])

def get_path(tree, index):
    if index == 0:
        return [], []
    for i in range(len(tree.tree_.children_left)):
        if tree.tree_.children_left[i] == index:
            path, truth_values = get_path(tree,i)
            path.append(str(i))
            truth_values.append(0)
            return path, truth_values
    for i in range(len(tree.tree_.children_right)):
        if tree.tree_.children_right[i] == index:
            path, truth_values = get_path(tree,i)
            path.append(str(i))
            truth_values.append(1)
            return path, truth_values
    return [], []

def get_parent(tree, index):
    if index == 0:
        return ""
    for i in range(len(tree.tree_.children_left)):
        if tree.tree_.children_left[i] == index:
            return str(i) + "_T"
    for i in range(len(tree.tree_.children_right)):
        if tree.tree_.children_right[i] == index:
            return str(i) + "_F"
    return ""

def maxval_feature(i, t, n, trees, items):
    return _LARGE_CONSTANT_

def node_lists(tree):
    ls = []
    ns = []
    for i in range(len(tree.tree_.value)):
        if tree.tree_.children_left[i] != -1:
            ns.append(i)
        else:
            ls.append(i)

    return ns, ls

def create_variables(tree):
    var_names = []
    var_types = ""
    var_lb = []
    var_ub = []
    var_obj = []
    
    nodes, leafs = node_lists(tree)
    num_rows = auction_size()

    for row in range(num_rows):
        for index in range(num_rows):
            var_names.append("index_" + str(row) + "_" + str(index))
            var_types = var_types + "B"
            var_lb.append(0)
            var_ub.append(1)
            var_obj.append(0)

    for row in range(num_rows):
        var_names.append("startprice_" + str(row))
        var_types = var_types + "I"
        var_lb.append(0.4 * float(get_feature_value(row,"EstValue")))
        var_ub.append(1.0 * float(get_feature_value(row,"EstValue")))
        var_obj.append(0)

    for row in range(num_rows):
        for node in nodes:
            var_names.append("node_" + str(row) + "_" + str(node))
            var_types = var_types + "B"
            var_lb.append(0)
            var_ub.append(1)
            var_obj.append(0)

    for row in range(num_rows):
        for target in TARGETS:
            var_names.append("target_" + str(row) + "_" + str(target))
            var_types = var_types + "B"
            var_lb.append(0)
            var_ub.append(1)
            var_obj.append(float(target)) 
            #var_obj.append(float(target)*get_feature_value(row, "EstValue"))
    return var_names, var_types, var_lb, var_ub, var_obj

def create_rows(tree):
    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = ""

    nodes, leafs = node_lists(tree)
    num_rows = auction_size()
    num_targets = len(TARGETS)
    
    for row in range(num_rows):
        col_names = [("index_" + str(row) + "_" + str(index)) for index in range(num_rows)]
        col_values = [1 for index in range(num_rows)]
        
        row_names.append("row_index_" + str(row))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"

    for index in range(num_rows):
        col_names = [("index_" + str(row) + "_" + str(index)) for row in range(num_rows)]
        col_values = [1 for row in range(num_rows)]
        
        row_names.append("index_row_" + str(index))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"

    for row in range(num_rows):
        col_names = [("target_" + str(row) + "_" + str(target)) for target in TARGETS]
        col_values = [1 for target in TARGETS]
        
        row_names.append("target_" + str(row))
        row_values.append([col_names,col_values])
        row_right_sides.append(1)
        row_senses = row_senses + "E"

    #for l in leafs:
    #    path, truth_values = get_path(tree, l)
    #    print path, truth_values
    
    for l in leafs:
        path, truth_values = get_path(tree, l)
        target = get_leaf_prediction(tree, l)
        
        for row in range(num_rows):
            max_val = 0
            
            col_names = []
            col_values = []
            
            leaf_unreachable = False
            for n in range(len(path)):
#                cn, cv = calc_feature(tree,row,n)
                cn, cv = calc_feature(tree,row,int(path[n]))
                if cn == [] and cv != []:
                    constant = get_node_constant(tree, int(path[n]))
                    if cv[0] <= constant and truth_values[n] == 1 or cv[0] > constant and truth_values[n] == 0:
                        leaf_unreachable = True
                        break
                else:
                    if truth_values[n] == 0:
                        col_names.append("node_" + str(row) + "_" + path[n])
                        col_values.append(-1)
                    else:
                        col_names.append("node_" + str(row) + "_" + path[n])
                        col_values.append(1)
                        max_val = max_val + 1
        
            if leaf_unreachable:
                continue

            col_names.append("target_" + str(row) + "_" + str(target))
            col_values.append(-1)

            row_names.append("predict_" + str(row) + "_" + str(l))
            row_values.append([col_names,col_values])
            row_right_sides.append(max_val - 1)
            row_senses = row_senses + "L"

    return row_names, row_values, row_right_sides, row_senses

def create_indicator_rows(tree):
    eps=10e-29
    row_names = []
    row_values = []
    row_right_sides = []
    row_senses = []
    row_complements = []
    row_indicators = []

    nodes, leafs = node_lists(tree)
    num_rows = auction_size()
    num_targets = len(TARGETS)
    
    for n in nodes:
        feature = get_feature(tree, n)
        constant = get_node_constant(tree, n)
        
        for row in range(num_rows):
            cons = 0
            col_names, col_values = calc_feature(tree,row,n)
            
            #print row, calc_feature(tree,row,n)
            
            if col_names == [] and col_values != []:
                continue
                #cons = col_values[0]
                #col_values = []

            row_indicators.append("node_" + str(row) + "_" + str(n))
            row_complements.append(1)
                
            row_names.append("node_ub_" + str(row) + "_" + str(n))
            row_values.append([col_names,col_values])
            row_right_sides.append(constant-cons)
            row_senses.append("L")

            row_indicators.append("node_" + str(row) + "_" + str(n))
            row_complements.append(0)

            row_names.append("node_lb_" + str(row) + "_" + str(n))
            row_values.append([col_names,col_values])
            row_right_sides.append(constant-cons+eps)
            row_senses.append("G")

    return row_names, row_values, row_right_sides, row_senses, row_indicators, row_complements

def find_path(tree, i):
    if i == 0:
        return ""
    for j in range(len(tree.tree_.children_left)):
        if tree.tree_.children_left[j] == i:
            return find_path(tree,j) + " " + str(j) + "_T"
    for j in range(len(tree.tree_.children_right)):
        if tree.tree_.children_right[j] == i:
            return find_path(tree,j) + " " + str(j) + "_F"
        
def print_tree(tree):
    for i in range(len(tree.tree_.value)):
        if tree.tree_.children_left[i] != -1:
            print(str(i)+  " decision node: feature "+str(FEATURE_MAP[tree.tree_.feature[i]]) + " <= " + str(tree.tree_.threshold[i]))
        else:
            print(str(i) + " leaf node: " + str(tree.tree_.value[i]) + " " + get_leaf_prediction(tree,i) + " path = (" + find_path(tree, i) + " )")

def lpdtree(tree, feasibility = 1):
    prob = cplex.Cplex()

    try:
        prob.objective.set_sense(prob.objective.sense.maximize)

        var_names, var_types, var_lb, var_ub, var_obj = create_variables(tree)
        prob.variables.add(obj = var_obj, lb = var_lb, ub = var_ub, types = var_types, names = var_names)

        row_names, row_values, row_right_sides, row_senses = create_rows(tree)
        prob.linear_constraints.add(lin_expr = row_values, senses = row_senses, rhs = row_right_sides, names = row_names)

        #prob.write("test.lp")

        row_names, row_values, row_right_sides, row_senses2, row_indicators, row_complements = create_indicator_rows(tree)
        for i in range(len(row_names)):
            prob.indicator_constraints.add(indvar = row_indicators[i], complemented = row_complements[i], rhs=row_right_sides[i], sense=row_senses2[i], lin_expr=row_values[i], name=row_names[i])

        #prob.write("test.lp")

        if feasibility == 1:
            prob.parameters.emphasis.mip.set(1) # focus of feasibility

        #prob.parameters.mip.polishing.time.set(5000)
        #prob.parameters.timelimit.set(5400)
        #prob.parameters.threads.set(1)

        prob.solve()
        
    except CplexError as exc:
        print(exc)
        return []
    
    # solution.get_status() returns an integer code
    #print("Solution status = " , prob.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(prob.solution.status[prob.solution.get_status()])

    if "infeasible" in prob.solution.status[prob.solution.get_status()]:
        return []
    
    print("Solution value  = ", prob.solution.get_objective_value())

    solution = []
    nodes, leafs = node_lists(tree)
    solution_values = prob.solution.get_values()
    num_rows = auction_size()
    for row in range(num_rows):
        #print solution_values[prob.variables.get_indices("startprice_" + str(row))]
        for i in range(len(FEATURE_NAMES)):
            if FEATURE_NAMES[i] == "StartPrice":
                OPTIMIZE_TABLE[row+1][i] = solution_values[prob.variables.get_indices("startprice_" + str(row))]
            if FEATURE_NAMES[i] == "SP.EV":
                OPTIMIZE_TABLE[row+1][i] = solution_values[prob.variables.get_indices("startprice_" + str(row))] / float(get_feature_value(row, "EstValue"))
        #values = [solution_values[prob.variables.get_indices("target_" + str(row) + "_" + str(target))] for target in TARGETS]
        #for i in range(len(values)):
        #    if values[i] == 1:
        #        print TARGETS[i]
        for index in range(num_rows):
            if solution_values[prob.variables.get_indices("index_" + str(row) + "_" + str(index))] == 1:
                for i in range(len(FEATURE_NAMES)):
                    if FEATURE_NAMES[i] == "LotNrRel":
                        OPTIMIZE_TABLE[row+1][i] = float(index) / float(num_rows)

    #print OPTIMIZE_TABLE

    dat = np.array(OPTIMIZE_TABLE)
    x = pd.DataFrame(data=dat[1:,0:])
    predictions = tree.predict(x)
    sum = 0
    #print predictions
    for i in predictions:
        sum += float(i)
    print("score check", sum)


              

# settings  
OUTPUT_DIR = './data_3/'
INPUT_DIR_1 = './data_3/'

test_nr = 3
depth_tree = 10         
read_file(INPUT_DIR_1 + "LP_tree_train_data.csv")  
read_optimize_file(INPUT_DIR_1 + "LP_tree_optimize_test_data_"+str(test_nr)+".csv")
the_tree = learn_tree(depth_tree)
print_tree(the_tree)
lpdtree(the_tree)



filename_OUT = "RESULT_DESIGN_v1_test_" + str(test_nr) #this file contains more info 
filename_OUT_pretty = "RESULT_DESIGN_v1_TBA_test_"  + str(test_nr) 
filename_FULL = 'LP_tree_FULL_test_data_' +str(test_nr)+".csv"

# run  post-processing on solution from MIP
feature_index = 4
feature_index_EV = 3
feature_index_SPEV = 9
lb_SPEV = 0.4
ub_SPEV = 1.0
feature_index_lotNr = 1
NEW_TABLE, OPT_TABLE_NEW = CODE_POST_PROCESS.apply_binary_search_MIP(DATA_TABLE, OPTIMIZE_TABLE, the_tree, feature_index, feature_index_EV , feature_index_SPEV,  lb_SPEV, ub_SPEV )
NEW_TABLE_sorted = CODE_POST_PROCESS.sort_EV_tree(OPT_TABLE_NEW, the_tree, feature_index_EV, feature_index_lotNr)

# write output to excel files
CODE_POST_PROCESS.make_results_TABLE_csv(OUTPUT_DIR, INPUT_DIR_1, NEW_TABLE, NEW_TABLE_sorted, filename_FULL, filename_OUT, filename_OUT_pretty)

