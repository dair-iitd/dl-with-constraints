#from IPython.core.debugger import Pdb
import time
import pickle 
import argparse
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import CnnEncoder


from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import SpanBasedF1Measure

from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from dataset import JointSeq2SeqDatasetReader
from models import LstmTagger
from allennlp.data.dataset import Batch
import os
import yaml
import torch.nn.functional as F
import utils
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from allennlp.nn import util as nn_util

from allennlp.models.archival import archive_model, load_archive, CONFIG_NAME

from allennlp.common.params import Params

from allennlp.training import util as training_util
from allennlp.common import util as common_util
import settings 



parser = argparse.ArgumentParser()
parser.add_argument('--ddlr',
                    help="lr used for dd", type=float,
                    default=0.05)

parser.add_argument('--dditer',
                    help="iterations  used for dd", type=int,
                    default=25)

parser.add_argument('--constraints_path',
                    help="yaml file containing constraints", type=str,
                    default='constraints.yml')

parser.add_argument('--exp_dirs',
                    help="directories containing experiement logs", nargs = '+'
                    )

parser.add_argument('--output_file',
                    help="file name where output is stored", type=str,
                    default = '../results/latest.csv'
                    )


parser.add_argument('--test',
                    help="run on test data if 1 else on val data", type=int,
                    default=0)

parser.add_argument('--collate',
                    help="should collate all the results?", type=int,
                    default=0)

args = parser.parse_args()
args.best_exp_id_file = args.output_file+'_{}_{}.csv'.format(args.ddlr,args.dditer)
args.output_file += '{}_{}_test_{}.csv'.format(args.ddlr, args.dditer, args.test)
gresults ={}
inf_count = 0
nn_time = 0
inf_time = 0
#print(yatin)
#output_file = '../results/ner-pos-bs8-cg-results.csv'
with torch.no_grad():
    #results_dirlist = ['../logs/ner-pos-bs8-cg-cw0', '../logs/ner-pos-bs8-cg-cw1']
    #Pdb().set_trace()
    results_dirlist = args.exp_dirs
    for result_dir in results_dirlist:
        pickle_file = os.path.join(result_dir, 'inference_lr{}_iter{}_test{}.pkl'.format(args.ddlr, args.dditer, args.test))
        print("Results will be saved here: ",pickle_file)
        if os.path.exists(pickle_file):
            lresults = pickle.load(open(pickle_file,'rb'))
            print("Pickle file exists. Loading {}".format(pickle_file))
        else:
            lresults = {}

        shuffle_dirs = os.listdir(result_dir)
        for i,shuffle_dir in enumerate(shuffle_dirs):
            shuffle_dir_path = os.path.join(result_dir, shuffle_dir)
            if os.path.isdir(shuffle_dir_path):
                train_sizes = os.listdir(shuffle_dir_path)
                #train_sizes = ['ts100']
                for train_size in train_sizes:
                    exp_dir = os.path.join(shuffle_dir_path, train_size)
                    print(exp_dir)
                    config_file = os.path.join(exp_dir,CONFIG_NAME)
                    if os.path.exists(config_file) and exp_dir not in lresults and os.path.exists(os.path.join(exp_dir,'best.th')):
                        #print("FOUND ", config_file)
                        params = Params.from_file(config_file)
                        settings.set_settings(params)
                        archive = load_archive(exp_dir, cuda_device = params['cuda_device'], weights_file = os.path.join(exp_dir,'best.th')) 
                        model = archive.model
                        params = archive.config 
                        validation_iterator_params = params["validation_iterator"]
                        validation_iterator = DataIterator.from_params(validation_iterator_params)
                        validation_iterator.index_with(model.vocab)
                        validation_and_test_dataset_reader = DatasetReader.from_params(params['validation_dataset_reader'])
                        validation_data_path = params['validation_data_path']
                        
                        if args.test == 1:
                            validation_data_path = os.path.join(os.path.dirname(params['validation_data_path']), params['test_data_file'])
                            print('args.test=1. test file: {}'.format(validation_data_path))
                        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
                        val_generator = validation_iterator(validation_data,
                                            num_epochs=1,
                                            shuffle=False)
                        #
                        #ner_coefs, pos_coefs = utils.get_dd_coefs(params['model']['constraints_path'], model.vocab)
                        ner_coefs, pos_coefs = utils.get_dd_coefs(params['dd_constraints']['config']['mtl_implication']['constraints_path'], model.vocab)
                        #
                        ner_metric = [SpanBasedF1Measure(model.vocab, tag_namespace="task1_labels", ignore_classes=
                                                ['O']) for _ in range(args.dditer+1)]

                        pos_metric = [CategoricalAccuracy() for _ in range(args.dditer+1)]
                        global_num_violations = torch.zeros(args.dditer+1)
                        total_words = 0
                        model.eval()
                        
                        for (i,batch) in enumerate(val_generator):
                            if i%10 == 0:
                                print('batch#',i)
                            #
                            if settings.cuda:
                                batch = nn_util.move_to_device(batch,0)
                            #
                            mask = get_text_field_mask(batch['sentence'])
                            c1 = time.time()
                            scores = model(**batch)
                            c2 = time.time()
                            ner_labels = batch['task1_labels']
                            pos_labels = batch['task2_labels']
                            ner_prob = F.softmax(scores['task1_tag_logits'], dim=-1)
                            pos_prob = F.softmax(scores['task2_tag_logits'], dim=-1)
                            ner_prob1, pos_prob1, num_iter, num_violations = utils.dd_inference(
                            ner_prob, pos_prob, mask, ner_coefs, pos_coefs, args.ddlr,
                            args.dditer, ner_metric, pos_metric, ner_labels,
                            pos_labels)
                            c3 = time.time()
                            inf_time += c3 - c2
                            nn_time += c2 - c1
                            inf_count += 1
                            global_num_violations += num_violations
                            total_words += mask.sum().item()
                        #
                        lresults[exp_dir] = torch.zeros(args.dditer+1,8)
                        for i in range(args.dditer+1):
                            nm = ner_metric[i].get_metric(True)
                            pm = pos_metric[i].get_metric(True)
                            lresults[exp_dir][i]= torch.Tensor([params['train_size'], 
                                                        params['constraints_wt'], 
                                                        params['shuffle_id'],i,
                                                        nm['f1-measure-overall'],pm,
                                                        global_num_violations[i], total_words])
                            #print(i,nm['f1-measure-overall'],pm)
                    else:
                        print("@@@@@@@ MISSING or skip @@@@@@", config_file)
        #
        print("Pickle the results for later use")
        pickle.dump(lresults, open(pickle_file,'wb'))
        gresults.update(lresults) 



if args.collate > 0:
    print("Collate everything in a table")
    exp_id_to_dir = {}
    table = None 
    
    for k,v in gresults.items():
        this_table =pd.DataFrame(data=v.numpy(), columns = ['ts','cw','sid', 'ni','f','a','nv','nw'])
        this_table['exp_id'] = k.split('/')[-3] 
        exp_id_to_dir[k.split('/')[-3]] = os.path.dirname(os.path.dirname(k)) 
        if table is  None:
            table = this_table
        else:
            table = table.append(this_table)
    
    table.to_csv(args.output_file)
    
    stats = ['mean','median','min','max','std']
    
    summ = table.groupby(['exp_id','ts','ni']).agg({'f': stats,'a':stats, 'nv': stats, 'nw': ['mean','count']})
    summ = summ.reset_index()
    summ.columns = ['_'.join(col).strip('_') for col in summ.columns.values]
    summ.to_csv(args.output_file + '-summary.csv')
    ## extract data that we need to plot
    # table with rows = ts, columns  = cw, 
    #col_name = 'nv_median'
    def extract_col(col_name, summ,niter = 15,file_handle = None,header='',subtract_col_name=None,get_best_of = None):
        staru = summ[summ.ni == 0].pivot_table(values = col_name, index='ts',columns='exp_id')
        exclude_from_diff = []
        if isinstance(get_best_of,pd.DataFrame):
            staru = pd.merge(staru,  get_best_of,left_on = ['ts'], right_on = ['ts'])
            groups = list(get_best_of.columns)
            for this_group in groups:
                staru[this_group+'_lu'] = staru.lookup(staru.index,staru[this_group])
                exclude_from_diff.append(this_group)
        elif isinstance(get_best_of,dict):
            for key in get_best_of:
                staru[key+'_max'] = staru[get_best_of[key]].max(axis=1)
                staru[key+'_maxid'] = staru[get_best_of[key]].idxmax(axis=1)
                exclude_from_diff.append(key+'_maxid')
        #
        staru.columns = [str(x)+'U' for x in staru.columns]
        exclude_from_diff = [x+'U' for x in exclude_from_diff]

        if isinstance(niter,int):
            starc = summ[summ.ni == niter].pivot_table(values = col_name, index='ts',columns='exp_id')
            starc.columns = [str(x)+'C'+str(niter) for x in starc.columns]
        else:
            scols = ['ts','exp_id','ni']
            if col_name not in scols:
                scols.append(col_name)
            starc = pd.merge(niter[['ts','exp_id','ni','nruns']],summ[scols], left_on = ['ts','exp_id','ni'], right_on=['ts','exp_id','ni'], how = 'left').pivot_table(values=col_name, index='ts',columns='exp_id')
            starc.columns = [str(x)+'Cb' for x in starc.columns]
    
        cucc= staru.join(starc)
        if subtract_col_name is not None:
            diff_table = cucc.drop(exclude_from_diff, axis= 1).subtract(cucc[subtract_col_name], axis=0)
            diff_table.columns = ['DIFF_'+x for x in diff_table.columns]
            cucc = cucc.join(diff_table)
        # 
        cucc = cucc.reset_index()
        if file_handle is not None:
            print(header, file =file_handle)
            cucc.to_csv(file_handle,index=False)
        #
        return cucc
    
    
    def subtract_col_num(df, col_num):
        return df.subtract(df[df.columns[col_num]])
    
    
    exp_id_to_best_iter  = {}
    if args.test == 0:
        #its validation. Please save where you get best performance
        iter_with_best_f = summ.loc[summ.groupby(['exp_id','ts']).f_mean.idxmax()]
        for this_exp_id in iter_with_best_f.exp_id.unique():
            this_iter_with_best_f = iter_with_best_f[iter_with_best_f['exp_id'] == this_exp_id]
            save_to = os.path.join(exp_id_to_dir[this_exp_id], 'best_iter_from_val_lr{}_iter{}.csv'.format(args.ddlr, args.dditer, args.test))
            this_iter_with_best_f = this_iter_with_best_f[['ts','exp_id','ni','nw_count']].reset_index(drop=True)
            this_iter_with_best_f.columns = ['ts','exp_id','ni','nruns']
            this_iter_with_best_f.to_csv(save_to)
            exp_id_to_best_iter[this_exp_id] = this_iter_with_best_f 
    else:
        for this_exp_id in summ.exp_id.unique():
            read_from = os.path.join(exp_id_to_dir[this_exp_id], 'best_iter_from_val_lr{}_iter{}.csv'.format(args.ddlr, args.dditer, 0))
            exp_id_to_best_iter[this_exp_id] = pd.read_csv(read_from,index_col = 0)
    
    
    
    exp_id_to_best_iter_table = pd.concat(list(exp_id_to_best_iter.values())).reset_index(drop=True)      
    
    
    niter = 'best'
    
    fh = open(args.output_file+'-plot-summary_niter{}.csv'.format(niter),'w')
    print("#Batches,{}".format(inf_count), file = fh)
    print("#Inference Time,{}".format(inf_time), file = fh)
    print("#DD Iterations,{}".format(args.dditer), file = fh)
    
    print("#NN Time,{}".format(nn_time), file = fh)
    
    if isinstance(exp_id_to_best_iter_table,int):
        print("#DD Iterations Results,{}".format(exp_id_to_best_iter_table), file = fh)
    else:
        print("#DD Iterations Results,{Dynamic-chosen from val}", file = fh)
    
    print('Median F Scores', file = fh)
    
    base_exp_id = 'ner-pos-gan-bs_8-sumpen-semi_false-wm_cm-ddoptim_sgd-ddlr_0-ifa_1-ifb_1-decay_0-cw_1'

    cl_list = [ 'ner-pos-gan-bs_8-sumpen-semi_false-wm_cm-ddoptim_sgd-ddlr_0.01-ifa_1-ifb_5-decay_0-cw_1',
            'ner-pos-gan-bs_8-sumpen-semi_false-wm_cm-ddoptim_sgd-ddlr_0.01-ifa_1-ifb_1-decay_0-cw_1',    
            'ner-pos-gan-bs_8-sumpen-semi_false-wm_cm-ddoptim_sgd-ddlr_0.050000000000000003-ifa_1-ifb_1-decay_0-cw_1',
                'ner-pos-gan-bs_8-sumpen-semi_false-wm_cm-ddoptim_sgd-ddlr_0.050000000000000003-ifa_1-ifb_5-decay_0-cw_1'
                ]
    
    slmin50p_list = [   'ner-pos-gan-bs_8-sumpen-minpct_50-semi_true-wm_cm-ddoptim_sgd-ddlr_0.01-ifa_1-ifb_5-decay_0-cw_1',
            'ner-pos-gan-bs_8-sumpen-minpct_50-semi_true-wm_cm-ddoptim_sgd-ddlr_0.01-ifa_1-ifb_1-decay_0-cw_1',            
            'ner-pos-gan-bs_8-sumpen-minpct_50-semi_true-wm_cm-ddoptim_sgd-ddlr_0.050000000000000003-ifa_1-ifb_1-decay_0-cw_1',
                        'ner-pos-gan-bs_8-sumpen-minpct_50-semi_true-wm_cm-ddoptim_sgd-ddlr_0.050000000000000003-ifa_1-ifb_5-decay_0-cw_1'
            ]

    get_best_of = {'CL': cl_list,'SL': sl_list, 'SL50p': slmin50p_list}
    #get_best_of = {'CL': cl_list}
    save_to = args.best_exp_id_file
   
    if args.test ==  1:
        get_best_of  = pd.read_csv(save_to,index_col = 0) 
        get_best_of = get_best_of.set_index('ts')

    mean_fscore = extract_col('f_mean',summ, exp_id_to_best_iter_table, fh, 'NER Mean F Scores',base_exp_id+'U',get_best_of = get_best_of)
    if args.test == 0:
        best_column_groups = mean_fscore[['ts'] + [x+'_maxidU' for x in get_best_of.keys()]]
        best_column_groups.to_csv(save_to)
    
    
    med_fscore = extract_col('f_median',summ, exp_id_to_best_iter_table, fh, 'NER Median F Scores',base_exp_id+'U')
    std_fscore = extract_col('f_std', summ, exp_id_to_best_iter_table, fh, 'NER STD of F Scores',get_best_of = get_best_of)
    
    nv = extract_col('nv_mean', summ, exp_id_to_best_iter_table, fh, 'Mean Num Violations',get_best_of = get_best_of)
    nw = extract_col('nw_mean', summ, exp_id_to_best_iter_table, fh, 'Total Words',get_best_of = get_best_of)
    nruns = extract_col('nw_count', summ, exp_id_to_best_iter_table, fh, 'Total Runs for this Experiment',get_best_of = get_best_of)
    dd_iterations = extract_col('ni',summ,exp_id_to_best_iter_table, fh, '#Iterations of DD Inference',get_best_of = get_best_of)
    
    med_acc = extract_col('a_median',summ, exp_id_to_best_iter_table, fh, 'POS Median Accuracy',base_exp_id+'U')
    mean_acc = extract_col('a_mean',summ, exp_id_to_best_iter_table, fh, 'POS Mean Accuracy')
    std_acc = extract_col('a_std', summ, exp_id_to_best_iter_table, fh, 'POS STD of Accuracy',base_exp_id+'U')
    
    fh.close()
    
    dd_infer_nv = summ.pivot_table(index='ni', columns=['exp_id','ts'],values='nv_mean',aggfunc= 'mean')
    dd_infer_f = summ.pivot_table(index='ni', columns=['exp_id','ts'],values='f_median',aggfunc= 'mean')
    dd_infer_acc = summ.pivot_table(index='ni', columns=['exp_id','ts'],values='a_median',aggfunc= 'mean')
    
    infer_gain_f = dd_infer_f - dd_infer_f.loc[0]
    infer_gain_acc = dd_infer_acc - dd_infer_acc.loc[0]
    
    fh = open(args.output_file+'-plot-inference.csv','w')
    print("NUM VIOLATIONS INFERENCE",file=fh)
    dd_infer_nv.to_csv(fh)
    print("NER INFERENCE",file=fh)
    infer_gain_f.to_csv(fh)
    print('\nPOS INFERENCE',file=fh)
    infer_gain_acc.to_csv(fh)
    fh.close()
    
    #Std dev of the gain and not of performance.
    
    #base_exp_id = 'ner-pos-gan-bs_8-semi_false-wm_cm-cw_0'
    baseline = table[(table['ni'] == 0) & (table['exp_id'] == base_exp_id)]
    baseline = baseline[['ts','sid','f','a','nv','exp_id']]
    mtable = pd.merge(table,baseline, how = 'left',on = ['ts','sid'], suffixes=('','_bl'))
    mtable['fgain'] = mtable['f'] - mtable['f_bl']
    mtable['again']  = mtable['a'] - mtable['a_bl']
    summ = mtable.groupby(['exp_id','ts','ni']).agg({'f': stats,'a':stats, 'nv': stats, 'nw': ['mean','count'], 'fgain': stats, 'again': stats})
    summ = summ.reset_index()
    summ.columns = ['_'.join(col).strip('_') for col in summ.columns.values]
    fgain_mean = extract_col('fgain_mean',summ,exp_id_to_best_iter_table,None)
    fgain_std = extract_col('fgain_std',summ,exp_id_to_best_iter_table,None)
    
    fh = open(args.output_file+'-plot-summary_niter{}.csv'.format(niter),'a') 
    extract_col('fgain_mean', summ, exp_id_to_best_iter_table, fh, 'Mean of F score Gain',get_best_of = get_best_of)
    extract_col('fgain_std', summ, exp_id_to_best_iter_table, fh, 'Std dev  of F score Gain',get_best_of = get_best_of)
    fh.close()
    
    """
    pp = PdfPages('plot_f_nv_lr{}_niter{}_wm{}.pdf'.format(args.ddlr,args.dditer,args.which_model))
    #plt.subplot(nrows=2,ncols=2)
    
    table = None 
    for k,v in results.items():
        this_table =pd.DataFrame(data=v.numpy(), columns = ['ts','ni','f','a','nv','nw'])
        this_table['exp_id'] = k
        if table is  None:
            table = this_table
        else:
            table = table.append(this_table)
    
    
    
    unique_ts = table['ts'].unique()
    
    for ts in unique_ts:
        df = table[table['ts'] == ts]
        fig = plt.figure()
        df.f.plot()
        df.nv.plot(secondary_y=True,style = 'g')
        df.plot(x='ni',y='f')
        df.plot(x='ni',y='nv',secondary_y=True,style = 'g')
        plt.title('ts={},nw:{} f score (left, blue), #viloations (right)'.format(ts,df.nw.iloc[0]))
        #plt.title('ts={} f score (left, blue), #viloations (right)'.format(ts))
        pp.savefig(fig)
        plt.close()
    
    pp.close()
    """
