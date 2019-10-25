
import argparse
import subprocess
import os
import pandas as pd
import pickle
from tqdm import tqdm
from sys import argv
import json
from collections import OrderedDict
import math


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive_dir', type=str)
    parser.add_argument('--is_parent', type=int, default=1)
    parser.add_argument('--out_file', type=str, default='get_violations')
    parser.add_argument('--force', type=int, default=1)
    parser.add_argument('--parts', type=int, default=1,
                        help='in how many part is the job divided?')
    parser.add_argument('--my_part', type=int, default=0,
                        help='which part am i running?')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size')
   
    parser.add_argument('--valid_data',type=int,default = 0, help='evaluate on test data or validation data?')
    parser.add_argument('--read_only', type=int, default=0)
    args = parser.parse_args()
    assert args.my_part < args.parts
    exclude_list = []
    if args.force:
        print("Force is strong!")
    #
    args.archives_path = args.archive_dir
    if args.is_parent:
        archive_dirs = os.listdir(args.archives_path)
        print('len(archive_dirs)', len(archive_dirs))
        archive_dirs = [x for x in archive_dirs if x not in exclude_list]
        print('After filtering len(archive_dirs)', len(archive_dirs))
        archive_dirs.sort()
        n = len(archive_dirs)
        each_size = math.ceil(n/args.parts)
        archive_dirs = archive_dirs[each_size *
                                    args.my_part: each_size*(args.my_part+1)]
        print('Selecting from : {} to {} , out of {}. My len: {}'.format(each_size*args.my_part, each_size*(args.my_part+1), n, len(archive_dirs)))
        args.archive_dirs = archive_dirs
    else:
        args.archive_dirs = [args.archives_path]
        args.archives_path = os.path.dirname(args.archives_path)

    return args


def evaluate(archive_dir, archives_path, batch_size=32, valid_data=0, read_only=0, force=False):

    this_dir = os.path.join(archives_path, archive_dir)
    max_metric = 0
    for i in range(1000):
        if os.path.exists(os.path.join(archives_path, archive_dir)+"/metrics_epoch_"+str(i)+".json"):
            max_metric = i
        else:
            break

    with open(os.path.join(archives_path, archive_dir)+"/metrics_epoch_"+str(max_metric)+".json") as f:
        js = json.load(f)
    #
    #metrics = pd.DataFrame.from_dict(js,orient="index")
    metrics = js
    #print(metrics)
    nodec_output_f, dec_output_f = -1,-1
    nodec_violations, dec_violations = -1,-1 
    nodec_nsentences, nodec_npropositions, dec_nsentences, dec_npropositions = -1,-1,-1,-1


    # running custom evaluator

    suffix = '_valset' if valid_data else ''
    cmd_suffix = ' --valid_data ' if valid_data else ' '
    pred_out = os.path.join(archives_path, archive_dir,
                            'eval_custom_pred_{}{}'.format(max_metric,suffix))
    gold_out = os.path.join(archives_path, archive_dir,
                            'eval_custom_gold_{}{}'.format(max_metric,suffix))


    dec_pred_out = pred_out + "_INFDEC"
    dec_gold_out = gold_out + "_INFDEC"
    
    nodec_output_file = os.path.join(
            archives_path, archive_dir, 'nodec_output_{}{}'.format(max_metric,suffix))
    dec_output_file = os.path.join(
            archives_path, archive_dir, 'dec_output_{}{}'.format(max_metric,suffix))
   
    violations_out_file = os.path.join(
        archives_path, archive_dir, 'get_violations_{}{}'.format(max_metric,suffix))
    
    print('Looking for: {}\n{}\n{}'.format(nodec_output_file,dec_output_file,violations_out_file))

    if not read_only:
        if force or (not os.path.exists(pred_out+'.txt')):
            cmd = "$HOME/anaconda3/bin/python eval_custom.py" + " --archive_dir " + \
                os.path.join(archives_path, archive_dir) + \
                " --pred_out "+pred_out + " --gold_out " + gold_out + " --batch_size " + str(batch_size) + cmd_suffix
            output = subprocess.check_output(cmd, shell=True)
            print(cmd)

        # running perl scripts

        if (force or
                (not os.path.exists(nodec_output_file))
                or
                (not os.path.exists(dec_output_file))):

            cmd1 = "./srl-eval.pl " + gold_out+".txt" + " " + \
                pred_out+".txt" + " > " + nodec_output_file
            cmd2 = "./srl-eval.pl " + dec_gold_out+".txt" + " " + \
                dec_pred_out+".txt" + " > " + dec_output_file
            subprocess.check_output(cmd1, shell=True)
            subprocess.check_output(cmd2, shell=True)
            print(cmd1)
            print(cmd2)

    if os.path.exists(nodec_output_file): 
        nodec_output = open(nodec_output_file, 'r').read()
        nodec_output_f = float(nodec_output.split("\n")[6].split()[-1])
        nodec_nsentences = float(nodec_output.split("\n")[0].split()[-1])
        nodec_npropositions = float(nodec_output.split("\n")[1].split()[-1])

    if os.path.exists(dec_output_file):
        dec_output = open(dec_output_file, 'r').read()
        dec_output_f = float(dec_output.split("\n")[6].split()[-1])
        dec_nsentences = float(dec_output.split("\n")[0].split()[-1])
        dec_npropositions = float(dec_output.split("\n")[1].split()[-1])

    # running get violations scripts
    
    if not read_only:
        if force or (not os.path.exists(violations_out_file + '_result_nondecode')):
            cmd3 = "$HOME/anaconda3/bin/python get_violations.py" + " --archive_dir " + \
                os.path.join(archives_path, archive_dir) + \
                " --out_file " + violations_out_file + " --batch_size " + str(batch_size) + cmd_suffix
            output = subprocess.check_output(cmd3, shell=True)
            print(cmd3)
            #print("python get_violations.py" + " --archive_dir "+ os.path.join(archives_path, archive_dir))

    # read output from get violations scripts
    if os.path.exists(violations_out_file+"_result_nondecode"):
        with open(violations_out_file+"_result_nondecode") as f:
            l = f.readlines()
            nodec_violations = l[2:3]+l[4:6]+l[7:10]+l[12:16]

    if os.path.exists(violations_out_file+"_result_decode"):
        with open(violations_out_file+"_result_decode") as f:
            l = f.readlines()
            dec_violations = l[2:3]+l[4:6]+l[7:10]+l[12:16]

    return nodec_output_f, dec_output_f, nodec_violations, dec_violations, metrics, nodec_nsentences, nodec_npropositions, dec_nsentences, dec_npropositions


def parse_output(data):
    retval = [('F1_non_dec', data[0])]
    
    if data[2] != -1:
        for sent in data[2]:
            retval.append(
                (sent.split(":")[0].strip()+'_nodec', float(sent.split(":")[-1].strip())))
    
    retval.append(('F1_dec', data[1]))
    
    if data[3] != -1:
        for sent in data[3]:
            retval.append(
                (sent.split(":")[0].strip() + '_dec', float(sent.split(":")[-1].strip())))
            # retval.append(float(sent.split(":")[-1].strip()))
    
    for temp_i, var_name in enumerate(['nodec_nsentences', 'nodec_npropositions', 'dec_nsentences', 'dec_npropositions']):
        retval.append([var_name,data[temp_i + 5]])

    retval = OrderedDict(retval)
    retval.update(data[4])
    
    #retval+= data[4]
    return retval


def main(args):
    check = []
    for dir_path in args.archive_dirs:
        print(dir_path)
        check.append(evaluate(dir_path, args.archives_path, args.batch_size, args.valid_data, read_only = args.read_only, force=args.force))
        # print(check[-1])
    #
    parsed_check = [parse_output(x) for x in check]
    #metrics = pd.DataFrame.from_dict(js,orient="index")
    data = pd.DataFrame(parsed_check)
    data.index = args.archive_dirs
    data = data.reset_index()
    data.to_csv(args.out_file)
    return data
    #data.columns = ["Model Name","F1_non_dec","span_max_non_dec","span_sum_LB_non_dec","trans_max_non_dec","Bterminal_non_dec","Lterminal_non_dec","num_violating_spans_non_dec","Total semantic spans_non_dec","Total semantic spans non id_non_dec","Total gold semantic spans_non_dec","Total gold semantic spans non id _non_dec","F1","span_max","span_sum_LB","trans_max","Bterminal","Lterminal","num_violating_spans","Total semantic spans","Total semantic spans non id","Total gold semantic spans","Total gold semantic spans non id",'best_epoch', 'peak_cpu_memory_MB', 'peak_gpu_0_memory_MB','peak_gpu_1_memory_MB', 'training_duration', 'training_start_epoch','training_epochs', 'epoch', 'training_precision-overall','training_recall-overall', 'training_f1-measure-overall','training_loss', 'training_cpu_memory_MB', 'training_lb', 'training_ub','training_uloss', 'training_BL_nzl', 'training_BL_tl','training_BL_apl', 'training_LB_nzl', 'training_LB_tl','training_LB_apl', 'training_B_nzl', 'training_B_tl', 'training_B_apl','training_L_nzl', 'training_L_tl', 'training_L_apl','training_transition_nzl', 'training_transition_tl','training_transition_apl', 'training_gpu_0_memory_MB','training_gpu_1_memory_MB', 'validation_precision-overall','validation_recall-overall', 'validation_f1-measure-overall','validation_loss', 'best_validation_precision-overall','best_validation_recall-overall', 'best_validation_f1-measure-overall','best_validation_loss']


if __name__ == "__main__":
    args = read_args()
    data = main(args)
