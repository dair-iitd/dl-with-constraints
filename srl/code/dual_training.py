import sys, os
if os.path.exists('/home/yatin/shome'):
    sys.path.insert(0,'/home/yatin/shome/hpcphd/srl_new/allennlp')

if os.path.exists('/home/keshav/yhome'):
    sys.path.insert(0,'/home/keshav/yhome/hpcphd/srl_new/allennlp')

import settings
from allennlp.training.optimizers import Optimizer
from allennlp import run as allenrun
from IPython.core.debugger import Pdb
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
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

from srl_custom_dataset import CustomSrlReader
from srl_custom_model import CustomSemanticRoleLabeler
from srl_constraints_max_choice import SRLConstraintsCustomAggChoice 

from allennlp.data.dataset import Batch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from allennlp.nn import util as nn_util
import os
import datetime
import logging
import shutil
import time
import re
import datetime
import traceback
from allennlp.common.params import Params
import gan_trainer

from allennlp.training import util as training_util
from allennlp.common import util as common_util
import argparse
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
import allennlp

def read_args():
    print(torch.__version__)
    print(allennlp.__version__)
    print(allennlp.__path__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover',
                        help='recover from existing model', action='store_true')
    
    parser.add_argument('--force',
                        help='force start by deleting existing model', action='store_true')
    
    parser.add_argument('--params_file',
                        help='allennlp jsonnet parameter file', type=str, default = '')
    
    
    parser.add_argument('--weight_dir', default=None)
    parser.add_argument('--weight_file', default=None)
    parser.add_argument('--dd_file', default=None)
    parser.add_argument('--training_state_file', default=None)
    parser.add_argument('--debug',type=int,default=0)
    parser.add_argument('--shuffle', type=int, default=1)
    #parser.add_argument('--lamda_fix', type=str,  default='f')
    #parser.add_argument('--lamda_init', type=float, default=0.1)
    
    args = parser.parse_args()
    print("args:", args)
    return (args)

def main(args):
    params = Params.from_file(args.params_file)

    # print('Data seed:{}, Percent data: {}'.format(shuffle_id, train_size))
    settings.cuda = params['cuda_device'] != -1
    common_util.prepare_environment(params)
    
    serialization_dir = params['serialization_dir']
    training_util.create_serialization_dir(
        params, serialization_dir, args.recover, args.force)
    common_util.prepare_global_logging(serialization_dir, True)
    logging.info("torch version: {}, allennlp version: {}, allennlp path: {}".format(torch.__version__, allennlp.__version__, allennlp.__path__))
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    semi_supervision = params.get('semi_supervised',False)
    which_mixer = params.get('which_mixer','cm')
    dd_warmup_iters = params.pop('dd_warmup_iters',1)
    dd_semi_warmup_iters = params.pop('dd_semi_warmup_iters',1)
    dd_update_freq = params.pop('dd_update_freq',2)
    constraints_wt = params.get('constraints_wt',0)
    calc_valid_freq = params.get('calc_valid_freq', 1)
    backprop_after_xbatches = params.pop('backprop_after_xbatches',1)
    min_pct_of_unlabelled = params.pop('min_pct_of_unlabelled', 0.0)
    dd_increase_freq_after = params.pop('dd_increase_freq_after', 0)
    dd_increase_freq_by = params.pop('dd_increase_freq_by', 0)
    dd_decay_lr = params.pop('dd_decay_lr',0)
    dd_decay_lr_after = params.pop('dd_decay_lr_after',1.0)
    grad_norm_before_warmup = params.pop('grad_norm_before_warmup',0)
    if semi_supervision:
        print("Semi Supervision On")
    
        
    for key in ['warmup_epochs','unlabelled_train_data_file','test_data_file', 'data_dir', 'cuda_device', 'serialization_dir', 'train_data_file', 'validation_data_file', 'constraints_wt', 'train_size', 'shuffle_id','semi_supervised','which_mixer','distributed_lambda_update', 'calc_valid_freq']:
        params.pop(key,None)
    
    print("Trainer pieces")
    pieces = gan_trainer.TrainerPiecesForSemi.from_params(
        params, serialization_dir, args.recover, semi_supervision)  # pylint: disable=no-member
    
    #pieces for constrained learning"
    print("Constraint model")
    constraints_model = Model.from_params(vocab=pieces.model.vocab, params=params.pop('dd_constraints'))
    dd_params = [[n, p] for n, p in constraints_model.named_parameters() if p.requires_grad]
    dd_optimizer = None
    dd_optim_params = params.pop('dd_optimizer',None)
    if len(dd_params) > 0:
        dd_optimizer = Optimizer.from_params(dd_params, dd_optim_params)
        
    cp = None
    chfile=None
    #Pdb().set_trace()
    if args.weight_dir is not None:
        #Pdb().set_trace()
        flag = True
        if args.weight_file is not None:
            logging.info("Loading  Model weights from :{}".format(os.path.join(args.weight_dir, args.weight_file)))
            model_states = torch.load(os.path.join(args.weight_dir, args.weight_file))
            pieces.model.load_state_dict(model_states)
            flag = False 
        if args.dd_file is not None:
            logging.info("Loading Constraint Model from :{}".format(os.path.join(args.weight_dir, args.dd_file)))
            flag = False 
            chfile = os.path.join(args.weight_dir, args.dd_file)
            # cp = torch.load(chfile)
            # constraints_model.load_state_dict(cp['constraints_model'])
            # if 'dd_update_freq' in cp:
            #     dd_update_freq  = cp['dd_update_freq']
            #     print("New dd_update_freq:" , dd_update_freq)
    
        if flag:
            raise("why provide args.weight_dir? when both weight_file and dd_file are None")
    print("Trainer")
    trainer = Trainer.from_params(
        model=pieces.model,
        serialization_dir=serialization_dir,
        iterator=pieces.iterator,
        train_data=pieces.train_dataset,
        validation_data=pieces.validation_dataset,
        params=pieces.params,
        validation_iterator=pieces.validation_iterator)
    
    if args.weight_dir is not None and args.training_state_file is not None: 
        logging.info("Loading Training state from :{}".format(os.path.join(args.weight_dir, args.training_state_file)))
        training_state = torch.load(os.path.join(args.weight_dir, args.training_state_file))
        trainer.optimizer.load_state_dict(training_state["optimizer"])
        
    
    params.assert_empty('base train command')
    
    try:
        #if backprop_after_xbatches == 1:
        #    print("Training setup")
        #    semi_trainer= gan_trainer.SemiSupervisedTrainer(trainer, constraints_model, dd_optimizer, pieces.validation_iterator,  pieces.unlabelled_dataset, semi_supervision, which_mixer, dd_warmup_iters, dd_update_freq, constraints_wt, calc_valid_freq)
        #    print("Training start")
        #    metrics = semi_trainer.custom_train()
        #else:
        print("Training setup")
        semi_trainer= gan_trainer.SemiSupervisedTrainer(trainer, constraints_model, dd_optimizer, 
                            pieces.validation_iterator,  pieces.unlabelled_dataset, semi_supervision, which_mixer, 
                            dd_warmup_iters, dd_update_freq, constraints_wt, calc_valid_freq, backprop_after_xbatches, 
                            min_pct_of_unlabelled,dd_semi_warmup_iters, dd_increase_freq_after, dd_increase_freq_by, 
                            dd_decay_lr, args.debug, chfile=chfile, shuffle=args.shuffle, dd_decay_lr_after = dd_decay_lr_after,
                            grad_norm_before_warmup=grad_norm_before_warmup)
        
    
        print("Training start")
        #print(yatin)
        metrics = semi_trainer.custom_train()
    
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir,
                          files_to_archive=params.files_to_archive)
        raise
    
    archive_model(serialization_dir,
                  files_to_archive=params.files_to_archive)
    common_util.dump_metrics(os.path.join(
        serialization_dir, "metrics.json"), metrics, log=True)




if __name__ == "__main__":
    args = read_args()
    main(args)

