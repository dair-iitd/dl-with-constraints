from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import math
import sys
import argparse
import os
import csv 
from general_utils import *


def read_multi_table(stats_file,colname):
    alines = open(stats_file).readlines()
    data  = []
    header = None
    gtab = None
    for i,acols in enumerate(csv.reader(alines)):
        #acols = l.strip().split(',')
        if colname not in acols:
            data.append(acols)
        #
        if colname in acols or i == (len(alines)-1):
            if header is not None and len(data) > 0:
                tab1 = pd.DataFrame(data,columns = header)
                data = []
                if gtab is None:
                    gtab = tab1
                else:
                    gtab = gtab.append(tab1)
            #
            header = acols
    return gtab


def read_table(stats_file,exp, non_numeric_cols = ['exp']):
    #if len(open(stats_file).readline()) == 0:
    #    return None
    table = read_multi_table(stats_file,non_numeric_cols[0])
    if table is None:
        return table

    table = table[table[table.columns[0]] != table.columns[0]]
    #if 'niter' in table.columns:
    #    table = table[table['niter'] != 'niter']
    numeric_cols = list(set(table.columns).difference(set(non_numeric_cols)))
    table[numeric_cols] = table[numeric_cols].apply(pd.to_numeric)
    params = exp.split('_')
    params = [p for p in params if p.find('-') != -1]
    param_name = [p.split('-')[0] for p in params]
    param_val = ['-'.join(p.split('-')[1:]) for p in params]
    for pname, pval in zip(param_name,param_val):
        try:
            pval = float(pval)
        except:
            pass
        #
        table[pname] = pval
    #
    return table

def collate_tables(input_dir, file_name,non_numeric_cols):
    table = None
    missing = []
    exps = os.listdir(input_dir)
    for this_exp in exps:
        logs_file = os.path.join(input_dir, this_exp,file_name)
        if os.path.exists(logs_file):
            this_table = read_table(logs_file,this_exp,non_numeric_cols)
        else:
            missing.append(this_exp)
            continue
        #
        if this_table is None or this_table.shape[0] < 5:
            missing.append(this_exp)
        else:
            if 'exp' not in this_table.columns:
                this_table['exp'] = this_exp 
            if table is None:
                table = this_table 
            else:
                table = table.append(this_table)

    st = table.reset_index(drop=True)
    return (st,missing)


def select_best(st,exp_id,wrt,max_iter_list = None):
    result = None 
    if max_iter_list is None:
        max_iter_list = [st.niter.max()]
    for max_iters in max_iter_list:
        summ = st[st['niter'] <= max_iters]
        summ = summ.reset_index(drop=True)
        summ = summ.loc[summ.groupby(exp_id)[wrt].idxmax()]
        summ = summ.reset_index(drop=True)
        summ['max_iters'] = max_iters 
        if result is None:
            result = summ
        else:
            result = result.append(summ)
    
    result = result.reset_index(drop=True)
    return result


def count_gt(x,thr = 0.5):
    return (x > thr).sum()/ (1.0*len(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', type=str,required=True)
    parser.add_argument('-output_file', type=str,required=True)
    args = parser.parse_args()
    
    names = ['dd_constant_lambda', 'ddlr', 'struct_weight', 'dd_mutl_excl_wt', 'dd_implication_wt',
                    'dd_maxp_wt', 'semi_sup']
    short_names = ['conlam', 'ddlr', 'sw', 'dmutexwt', 'dimpwt',
                'dmaxpwt', 'semi']
#
    names = [n.replace('_','.') for n in names] 

    st,missing  = collate_tables(args.input_dir,'stats.csv',['exp'])
    result = select_best(st,'exp','dev_map')
    
    prob_stats,missing1 = collate_tables(args.input_dir,'test_results.csv_scores_sum_gold_count.csv',['entities'])
    #prob_stats,missing1 = collate_tables(args.input_dir,'test_sample_results.csv_scores_sum_gold_count.csv',['entities'])
    pstats = prob_stats.groupby(['exp']).agg({'ms': count_gt})
    
    result = result.join(pstats,on='exp')
    
    result.to_csv(args.output_file+'_best.csv')
    st.to_csv(args.output_file+'_raw.csv')
    aggcols = ['test_map','test_f_macro','test_r_macro','test_p_macro','test_f_micro','test_r_micro','test_p_micro']
    
    aggcols = ['dev_map','test_map','test_acc','test_f_macro','test_f_micro','test_r_micro','test_p_micro','test_r_macro','test_p_macro','niter','nliter','test_maxp_apl','test_imp_apl','test_me_apl','ms','test_me_pv','test_imp_pv','test_te']
    
    index = short_names
    index0 = index  
    index = [x for x in index0 if x in result.columns]
    missing = [x for x in index0 if x not in result.columns]
    print("MIssing indices. Ignoring...",missing)
    result[index] = result[index].fillna(-1)

    pivot = result.pivot_table(values = aggcols,index=index,columns=None, aggfunc=['mean'])
    pivot.to_csv(args.output_file + '_pivot.csv')
    
    st = st.reset_index(drop=True)
    exp_list = result.groupby(index).agg({'exp': 'first'})['exp']
    
    pp = PdfPages(args.output_file+ '_plots.pdf')
    plt.rcParams.update({'axes.titlesize': 'small'})
    for this_exp in exp_list:
        tst = st[st['exp'] == this_exp]
        tst = tst.set_index('niter')
        fig = plt.figure()
        tst.dev_map.plot()
        tst.lr.plot(secondary_y=True,style='g')
        plt.title(split_title_line(' '.join([x+'-'+str(tst.iloc[0][x]) for x in index]),max_words=5))
        pp.savefig(fig)
        plt.close()
    
    pp.close()
   
