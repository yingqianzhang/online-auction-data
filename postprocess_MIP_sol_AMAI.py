# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:32:06 2019

@author: jrhuggenaa
"""

from sklearn import tree

import numpy as np
import pandas as pd
import csv

# function reading in data
#Reads the <inputfile>.csv and makes DataFrame df
def read_data(inputfile):
    df = pd.read_csv(inputfile)
    return df

#Reads the <inputfile>.csv and makes DataFrame df with delimiter ';'
def read_data_semicolon(inputfile):
    df = pd.read_csv(inputfile, sep=';')
    return df

def read_data_comma(inputfile):
    df = pd.read_csv(inputfile, sep=',')
    return df

#Reads the <inputfile>.csv and makes DataFrame df with delimiter ';'
def read_data_tab(inputfile):
    df = pd.read_csv(inputfile, sep='\t', header = None)
    return df

def filter_data(df, condition_col, condition_val):
    newdf = df.loc[df[condition_col] == condition_val]
    newdf.reset_index(drop=True, inplace=True)
    return newdf

#Selects lots with <col> in <val_list>
def filter_isin_list(df, col, val_list):
    newdf = df.loc[df[col].isin(val_list)]
    newdf.reset_index(drop=True, inplace=True)
    return newdf

def write_file(file_name,xData,yData):
    xData_copy = xData.copy()
    if yData is not None:
        xData_copy = xData_copy.assign(yData=yData)
    xData_copy.to_csv(file_name, sep=';', header=True, index=False)
    
# end functions reading in data
def roundColumn(allData, col, decimals):
    allData = allData.round({col: decimals})
    return allData;

#Returns list of unique items from <itemlist> in .keys(), with the count in .values(). Itemlist is e.g. df['colname']
def get_distribution(itemlist):
    distribution = dict()
    for j in range(len(itemlist)):
        if itemlist[j] not in distribution:
            distribution[itemlist[j]] = 1
        else:
            distribution[itemlist[j]] += 1
    return distribution
  
def selectData(xTrain, cols):
    xTrain = xTrain.ix[:,cols]

    return xTrain;

def mergeDataFrames(df1, df2, col):
    df_concat = pd.concat([df1, df2], axis=col)
    return df_concat;

#Add data in column <colGet> from <df2> to <df> with name <colName>, matching <key_df> from <df> with <key_df2> from <df2>
def set_external_col(df, key_df, colName, df2, key_df2, colGet):
    newdf = df.copy()
    df2new = df2.set_index(key_df2)[colGet].to_dict()
    newdf[colName] = newdf[key_df].map(df2new)
    return newdf

#Add new column by combining the contents of two columns
def add_combined_col(df, colName, col1, col2):
    keys = []
    for i in range(len(df)):
        keys.append(str(df[col1].iloc[i]) + '-' +  str(df[col2].iloc[i]))
    df2 = df.copy()
    df2[colName] = keys
    return df2


def BinarySearchTree(tree_clf, input_feature, class_label, var_index,feature_index_EV , feature_index_SPEV, lb, ub, start_val , abs_tol):
    """
    search for lower values of SP 
    """
    terminate = False
    lb_alg = lb
    ub_alg = ub
    sol_old = start_val  # set old solution (which is feasible)
    
    input_feature_new = list(input_feature)
    
    # get initial midpoint
    midpoint =   lb_alg*0.5 + start_val*0.5 
    # initialize current solution to old solution
    sol_current = sol_old
    iteration = 0
# while loop
    while terminate == False:    
        iteration = iteration + 1
        
        input_feature_new = get_new_feature_vector(input_feature_new, midpoint, var_index, feature_index_EV , feature_index_SPEV)
        new_label = list(tree_clf.predict([  input_feature_new ]) )
        new_label = new_label[0]
        
        # do test
        violation = check_ClassLabel_condition(class_label, new_label)
        
        if violation == 0:
            
            if iteration > 1:
                sol_old = sol_current

            sol_current = midpoint
            ub_alg = midpoint

        elif violation == 1:
            lb_alg = midpoint
            ub_alg = sol_current
              

        midpoint = calc_midpoint(lb_alg, ub_alg)
        
        if iteration > 1:
            stop = check_convergence_solution(sol_old, sol_current, abs_tol)
        else:
            stop = 0
        if stop == 1:
            terminate =True
    
    return sol_current ;


def calc_midpoint(x_min, x_max):
    
    midpoint = x_min*0.5 + x_max*0.5
    
    return midpoint ;

def check_ClassLabel_condition(target_label, new_label):
    
    if new_label >= target_label :
        violation = 0
    else :
        violation = 1
    
    
    return violation ;

def check_convergence_solution(sol_old, sol_new, abs_tol):
    
    if abs(sol_old-sol_new) < abs_tol:
        stop = 1
    else :
        stop = 0

    return stop ;

def get_new_feature_vector(old_feature_vector, new_SP, feature_index, feature_index_EV , feature_index_SPEV):
    
    new_feature_vector = list(old_feature_vector)
    
    new_feature_vector[feature_index - 1] = new_SP 
    EV = old_feature_vector[feature_index_EV - 1]
    new_feature_vector[feature_index_SPEV - 1]  =     float(new_SP)/ float(EV)
    
    return new_feature_vector ;

def get_bounds_SP(input_feature_vector, feature_index_EV, perc_lb, perc_ub):
    
    EV = input_feature_vector[feature_index_EV - 1]
    lb = float(EV*perc_lb)
    ub = float(EV*perc_ub)
    
    return lb, ub ; 

def make_modfied_LotNr(df):
    
    num_lots = len(df['SaleNumber'])
    df.rename(index=str, columns={'LotNr': 'LotNr_Original', }, inplace=True)
    
    df['LotNr'] = range(1, num_lots + 1 , 1)
    
    return df ; 

def sort_toIntegers(input_list):
    
    input_list_sorted = list(np.sort(input_list))
    
    lot_nr_list = range(1, len(input_list_sorted) + 1, 1)
    int_dict = make_empty_dict(input_list_sorted, lot_nr_list)
    
    output_list = []
    for k in input_list:
        val = int_dict[k]
        output_list.append( val )
    return output_list ;

def apply_binary_search_MIP(DATA_TABLE, OPTIMIZE_TABLE, tree_clf, feature_index, feature_index_EV , feature_index_SPEV, perc_lb, perc_ub):
    """
    search for lower values of SP such that perc_lb*EV<= SP <=perc_ub*EV
    """      
    
    abs_tol = 1.05

    new_values = []
    NEW_TABLE = []
    OPT_TABLE_NEW = []
    
    names = OPTIMIZE_TABLE[ 0 ] + [ 'OLD_SPEV' , 'OLD_SP', 'LB_SP_MIP' , 'UB_SP_MIP' , 'SP_NEW' , 'SPEV_NEW', 'yClass_pred', 'yClass_MIP' ]

    
    NEW_TABLE.append(names)
    OPT_TABLE_NEW.append( OPTIMIZE_TABLE[ 0 ]  )
    
    
    total_rows = len(OPTIMIZE_TABLE) 
    for i in range(2, total_rows + 1, 1):
        
        add_list  = []
        
        input_feature = OPTIMIZE_TABLE[i - 1]
        lb, ub = get_bounds_SP(input_feature, feature_index_EV, perc_lb, perc_ub)
        start_val = input_feature[feature_index - 1]
        
        add_list  =  add_list + input_feature
        add_list.append(input_feature[feature_index_SPEV - 1])
        add_list.append(input_feature[feature_index - 1])
        add_list.append(lb)
        add_list.append(ub)
        

        ub = min(start_val, ub)
        start_val = ub
        class_label = list(tree_clf.predict([  input_feature ]) )
        class_label = class_label[0]
        class_label_MIP_start = class_label[0]
        val = BinarySearchTree(tree_clf, input_feature, class_label, feature_index, feature_index_EV, feature_index_SPEV, lb, ub, start_val , abs_tol)
        new_values.append(val)

        input_feature_ORG = DATA_TABLE[i - 1]
        input_feature_ORG = input_feature_ORG[0: len(input_feature_ORG)- 1]
        
        class_label = list(tree_clf.predict([  input_feature_ORG ]) )
        class_label = float(class_label[0])

        input_feature_new = get_new_feature_vector(input_feature, val, feature_index, feature_index_EV , feature_index_SPEV)
        class_label_new = list(tree_clf.predict([  input_feature_new ]) )
        class_label_new = float(class_label_new[0])
        
        add_list.append(input_feature_new[feature_index - 1])
        add_list.append(input_feature_new[feature_index_SPEV - 1])
        add_list.append( class_label )
        add_list.append( class_label_new )
        #print("class old = " + str(class_label) + " Class new = " + str(class_label_new) + " Class MIPstart = " + str(class_label_MIP_start) )
        NEW_TABLE.append(add_list)
        OPT_TABLE_NEW.append( input_feature_new  )
        
    dat = np.array(NEW_TABLE)
    df = pd.DataFrame(data=dat[1:,0:] , columns= names)

    
    return df, OPT_TABLE_NEW ; 


def make_empty_dict(action_list, output_list):
    
    if output_list == None:
        output_list = [ []  ]*len(action_list)
    dict_out = dict(zip( action_list, output_list)) 
    
    return dict_out ;

def sort_EV_leafid(leaf_id, rowid_dict, EV_dict, lot_nr_dict ):
    rowid_list = rowid_dict[leaf_id]
    EV_list = EV_dict[leaf_id]
    lot_nr_list = lot_nr_dict[leaf_id]
    
    row_id_sort = []
    lot_nr_sort = list(np.sort(lot_nr_list))
    
    index_EV_sort = list(np.argsort(EV_list))
    index_EV_sort.reverse()
    
    for k in range(1, len(index_EV_sort) + 1, 1):
        index_EV = index_EV_sort[k-1]
        row_id_sort.append( rowid_list[index_EV]   )
    
    
    rowid_lotnr_dict = make_empty_dict(row_id_sort, lot_nr_sort)
    return rowid_lotnr_dict ;

def sort_EV_tree(OPT_TABLE_NEW, tree_clf, feature_index_EV, feature_index_lotNr):
    """
    re-order lot numbers in same leaf of tree so that lots with higher values of EV have lower lot numbers
    """    
    OPT_TABLE_NEW_data = OPT_TABLE_NEW[1:]
    leave_id = list(tree_clf.apply(OPT_TABLE_NEW_data))
    leave_id_unique = list(np.unique(leave_id))
    

    rowid_dict = make_empty_dict(leave_id_unique, None)
    EV_dict = make_empty_dict(leave_id_unique, None)
    lot_nr_dict = make_empty_dict(leave_id_unique, None)
    
    rowid_lotnr_dict_all = make_empty_dict(leave_id_unique, None)
    
    total_rows = len(OPT_TABLE_NEW_data) 
    for i in range(1, total_rows + 1, 1):
        input_feature = list(OPT_TABLE_NEW_data[i - 1])
        leaf_id = leave_id[i-1]
        
        EV_val = input_feature[feature_index_EV-1]
        row_id = i
        lot_nr = input_feature[feature_index_lotNr-1]
        

        rowid_dict[leaf_id] = list(rowid_dict[leaf_id]) + [row_id]

        EV_dict[leaf_id] = list(EV_dict[leaf_id]) + [EV_val]
        
      
        lot_nr_dict[leaf_id] = list(lot_nr_dict[leaf_id]) + [lot_nr]

    #print("lot_nr_dict = " + str(lot_nr_dict) ) 
    for k in leave_id_unique:
        rowid_lotnr_dict = sort_EV_leafid(k, rowid_dict, EV_dict, lot_nr_dict )
        rowid_lotnr_dict_all[k] = rowid_lotnr_dict

    OPT_TABLE_NEW_sortEV = []

    
    
    for i in range(1, total_rows + 1, 1):
        input_feature_new = list(OPT_TABLE_NEW_data[i - 1])
        input_feature_new2 = list(OPT_TABLE_NEW_data[i - 1])

        leaf_ID_check = list(tree_clf.apply([input_feature_new2]))
        leaf_ID_check = int(leaf_ID_check[0])

        y_pred_check =  list(tree_clf.predict([  input_feature_new2 ]) )
        y_pred_check = float(y_pred_check[0]) 
        
        leaf_id = leave_id[i-1]
        lot_nr_NEW =    rowid_lotnr_dict_all[leaf_id][i]     
        
        input_feature_new2[feature_index_lotNr-1] = lot_nr_NEW

        y_pred_sort =  list(tree_clf.predict([  input_feature_new2 ]) )
        y_pred_sort = float(y_pred_sort[0])         
        
        leaf_ID = list(tree_clf.apply([input_feature_new2]))
        leaf_ID = int(leaf_ID[0])
        input_feature_new =  input_feature_new +  [  lot_nr_NEW      ]  + [ y_pred_check ] + [ y_pred_sort ] + [ leaf_id ] + [ leaf_ID_check ] + [leaf_ID]
        OPT_TABLE_NEW_sortEV.append( input_feature_new )
    
    names = OPT_TABLE_NEW[ 0 ] + ['LotNrRel_NEW_sorted'] + ['y_pred_check']  + ['y_pred_sort'] + [ 'leaf_id'] + [ 'leaf_ID_check'] + ['leaf_ID_sort']
    dat = np.array(OPT_TABLE_NEW_sortEV)
    df = pd.DataFrame(data=dat  , columns= names)
    
    return df;

def make_results_TABLE_csv(OUTPUT_DIR, INPUT_DIR, NEW_TABLE, NEW_TABLE_sorted, filename_FULL, filename_OUT , filename_OUT_pretty = None):
    
    inputfile = INPUT_DIR + filename_FULL
    df = read_data_semicolon(inputfile)     

    cols = ['SaleNumber', 'LotNr', 'LotNrRel', 'EstValue', 'SP.EV', 'StartPrice' , 'yData' , 'Branch', 'MainCategory' , 'SubCategory' , 'SaleDate' , 'Status' , 'Mult']
    df = selectData(df, cols)
    df = make_modfied_LotNr(df)
    
    df.rename(index=str, columns={'SP.EV': 'SPEV' , 'StartPrice' : 'SP' , 'yData' :'yClass' }, inplace=True)

    cols = [ 'LotNrRel', 'SPEV_NEW', 'SP_NEW' , 'yClass_pred' , 'yClass_MIP']
    df2 = selectData(NEW_TABLE, cols)
    df2.rename(index=str, columns={ 'LotNrRel': 'LotNrRel_NEW'}, inplace=True)
 
	
    # make absolute LotNr
    num_row = float(len(df2['LotNrRel_NEW']))
    new_val = sort_toIntegers(list(df2['LotNrRel_NEW']))
    df2['LotNr_NEW']= new_val
    
    cols = ['LotNr_NEW', 'LotNrRel_NEW', 'SPEV_NEW', 'SP_NEW' , 'yClass_pred' , 'yClass_MIP']
    df2 = selectData(df2, cols)
    
    new_val = sort_toIntegers(list(NEW_TABLE_sorted['LotNrRel_NEW_sorted']))
    df2['LotNr_NEW_sorted']= new_val
    df2['LotNrRel_NEW_sorted']= list(NEW_TABLE_sorted['LotNrRel_NEW_sorted'])
       
    df['LotNr_NEW'] = df2['LotNr_NEW']
    df['LotNrRel_NEW'] = df2['LotNrRel_NEW']
    df['LotNr_NEW_sorted'] = df2['LotNr_NEW_sorted']
    df['LotNrRel_NEW_sorted'] = df2['LotNrRel_NEW_sorted']
    df['SPEV_NEW'] = df2['SPEV_NEW']
    df['SP_NEW'] = df2['SP_NEW']    
    df['yClass_pred'] = df2['yClass_pred']
    df['yClass_MIP'] = df2['yClass_MIP']
	
    if filename_OUT_pretty!= None:
        df.to_excel(OUTPUT_DIR+filename_OUT_pretty +'.xlsx', header=True, index=False)     
    df['y_pred_check'] = list(NEW_TABLE_sorted['y_pred_check'])    
    df['y_pred_sort'] = list(NEW_TABLE_sorted['y_pred_sort'])
    df['leaf_id'] = list(NEW_TABLE_sorted['leaf_id'])
    df['leaf_ID_check'] = list(NEW_TABLE_sorted['leaf_ID_check'])
    df['leaf_ID_sort'] = list(NEW_TABLE_sorted['leaf_ID_sort'])

    df.to_excel(OUTPUT_DIR+filename_OUT+'.xlsx', header=True, index=False) 
    
    #df.to_csv(filename_OUT+'.csv', sep=';', header=True, index=False) 
    
    
    return ; 

   

