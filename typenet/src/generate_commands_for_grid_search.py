
"""
HOW to RUN?

python generate_commands_for_grid_search.py -log_dir ../logs -base_dir ../resources -dataset typenet -take_frac 0.05 -file_name 5p -save_model 1 -num_streams 1
python generate_commands_for_grid_search.py -log_dir ../logs -base_dir ../resources -dataset typenet -take_frac 0.1 -file_name 10p -save_model 1 -num_streams 1
python generate_commands_for_grid_search.py -log_dir ../logs -base_dir ../resources -dataset typenet -take_frac 1.0 -file_name 100p -save_model 1 -num_streams 1

"""

from __future__ import print_function

import subprocess
import itertools
import argparse
import sys
import os
from time import sleep
import random
import stat
import copy
parser = argparse.ArgumentParser()
parser.add_argument('-log_dir', default='logs', required=True, type=str)
parser.add_argument('-base_dir', default='', required=True, type=str)
parser.add_argument('-dataset', default='typenet', type=str)
parser.add_argument('-take_frac', default=0.05, type=float)
parser.add_argument('-file_name', default='all_jobs.sh', type=str)
parser.add_argument('-save_model', default=0, type=int)
parser.add_argument('-num_streams', default=1, type=int)
args = parser.parse_args(sys.argv[1:])


log_dir = "%s_%5.4f" % (args.log_dir, args.take_frac)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists('TMP_LOGS'):
    print('Creating TMP_LOGS directory')
    os.makedirs('TMP_LOGS')

dairpy = '$HOME/anaconda3/envs/py2/bin/python'
dair_base_cmd = '%s deploy_mil_constraints.py -dataset typenet -base_dir %s -take_frac %5.4f -linker_weight 0.0 -clip_val 10 -batch_size 10 -mode typing -embedding_dim 300 -lr 0.001 -beta1 0.9 -beta2 0.999 -epsilon 1e-4 -typing_weight 1 -test_batch_size 20 -save_model %s -log_dir %s -dd -num_epochs 150 -weight_decay 1e-6 -asymmetric 0 -dd_update_freq 10 -dd_weight_decay 0.0 -dd_increase_freq_after 1' % (
    dairpy, args.base_dir, args.take_frac, str(args.save_model), log_dir)

encoder = ['basic']
asymmetric = [0]
dropout = [0.5]
complex = [0]
use_transitive = [0]
dd_optim = ['sgd']
dd_penalty = ['mix']
dd_mom = [0.0]
dd_logsig = [0]
unlabelled_ratio = [20]
semi_mixer = ['cm']
use_wt_as_lr_factor = [0]

dd_warmup_iter = [1000]
dd_increase_freq_by = [0]
grad_norm = [0.15]

all_params = [dd_optim, dd_mom, dd_logsig,  dd_penalty, dropout, complex, use_transitive, semi_mixer,
              encoder, use_wt_as_lr_factor,  dd_warmup_iter, dd_increase_freq_by, unlabelled_ratio, grad_norm]
names = ['dd_optim', 'dd_mom', 'dd_logsig', 'dd_penalty', 'dropout', 'complex', 'use_transitive', 'semi_mixer',
         'encoder', 'use_wt_as_lr_factor', 'dd_warmup_iter', 'dd_increase_freq_by', 'unlabelled_ratio', 'grad_norm']
short_names = ['do', 'dm', 'dls', 'dp', 'drop', 'c', 'ut',
               'smix', 'e', 'uwlr', 'warmup', 'ifb', 'ulrat', 'gn']
jobs = list(itertools.product(*all_params))

additional_names = ['dd_constant_lambda', 'ddlr', 'struct_weight', 'dd_mutl_excl_wt', 'dd_implication_wt',
                    'dd_maxp_wt', 'semi_sup']
short_names += ['conlam', 'ddlr', 'sw', 'dmutexwt', 'dimpwt',
                'dmaxpwt', 'semi']
# baseline, constraints, hierarchy, semi sup
additional_job_list = [
    # baseline
    (0, 0, 0, 0, 0, 0, 0),
    # constraint learning
    (0, 0.01, 0.0, 1.0, 1.0, 0.0, 0),
    (0, 0.02, 0.0, 1.0, 1.0, 0.0, 0),
    (0, 0.03, 0.0, 1.0, 1.0, 0.0, 0),
    (0, 0.04, 0.0, 1.0, 1.0, 0.0, 0),
    (0, 0.05, 0.0, 1.0, 1.0, 0.0, 0),
    # semi supervision
    (0, 0.01, 0.0, 1.0, 1.0, 0.0, 1),
    (0, 0.02, 0.0, 1.0, 1.0, 0.0, 1),
    (0, 0.03, 0.0, 1.0, 1.0, 0.0, 1),
    (0, 0.04, 0.0, 1.0, 1.0, 0.0, 1),
    (0, 0.05, 0.0, 1.0, 1.0, 0.0, 1),
    # constant lambda
    (1, 0.00, 0.0, 1.0, 1.0, 0.0, 0),
    (1, 0.00, 0.0, 0.1, 0.1, 0.0, 0),
    (1, 0.00, 0.0, 0.01, 0.01, 0.0, 0),
    (1, 0.00, 0.0, 0.001, 0.001, 0.0, 0),
    (1, 0.00, 0.0, 0.0001, 0.0001, 0.0, 0),
    (1, 0.00, 0.0, 0.00001, 0.00001, 0.0, 0),
    # struct weight
    (0, 0, 0.5, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0),
    (0, 0, 2.0, 0, 0, 0, 0),
    (0, 0, 4.0, 0, 0, 0, 0)
]

names = names + additional_names
all_jobs = list(itertools.product(jobs, additional_job_list))
jobs_list = {}
sorted_names = copy.deepcopy(names)
sorted_names.sort()
assert len(names) == len(short_names)
assert len(set(short_names)) == len(short_names)
names2short = dict(zip(names, short_names))
for i, setting in enumerate(all_jobs):
    setting = list(itertools.chain(*setting))
    name_setting = {n: s for n, s in zip(names, setting)}
    # remove redundant settings
    # remove redundant settings
    setting_list = ['-%s %s' % (name, str(value))
                    for name, value in name_setting.iteritems()]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (names2short[n].replace('_', '.'),
                                   str(name_setting[n])) for n in sorted_names])
    jobs_list[log_str] = setting_str

print('writing logs to %s' % ( log_dir))
print('Writing total of %s commands to %s files, i = 0 to %s' % (len(jobs_list),args.file_name+'_dair_i.sh',args.num_streams))

#fhdair = open(os.path.join(args.jobs_dir, args.file_name+'_dair.sh'),'w')
mode = stat.S_IROTH | stat.S_IRWXU | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP
counter = 0
fhdair_dict = {}
for i in range(args.num_streams):
    fhdair_dict[i] = open(args.file_name+'_dair_{}.sh'.format(i), 'w')

for log_str, setting_str in jobs_list.iteritems():
    print('CUDA_VISIBLE_DEVICES=0 {} {} -model_name {} > TMP_LOGS/{}_LOG_{} 2>&1'.format(dair_base_cmd,
                                                                                         setting_str, log_str, args.file_name, counter), file=fhdair_dict[counter % args.num_streams])
    counter += 1

for fhdair in fhdair_dict.values():
    fhdair.close()
    os.chmod(fhdair.name, mode)

