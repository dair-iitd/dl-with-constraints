import os
import sys
import time 

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
from allennlp.data.dataset import Batch
from allennlp.common.params import Params
from allennlp.training import util as training_util
from allennlp.common import util as common_util
import argparse
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS


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

import gan_trainer_hm
from dataset import JointSeq2SeqDatasetReader
from models import LstmTagger
from mtl_constraints import MTLConstraints  

parser = argparse.ArgumentParser()
parser.add_argument('--recover',
                    help='recover from existing model', action='store_true')

parser.add_argument('--force',
                    help='force start by deleting existing model', action='store_true')

parser.add_argument('--params_file',
                    help='allennlp jsonnet parameter file', type=str, default = '')

parser.add_argument('--train_size_list',
                    help="list of train sizes", nargs = '+', type = int
                    )

parser.add_argument('--shuffle_id_list',
                    help="list of shuffle ids", nargs = '+', type = int 
                    )
#parser.add_argument('--lamda_fix', type=str,  default='f')
#parser.add_argument('--lamda_init', type=float, default=0.1)

parser.add_argument('--time_file',help='file in which time taken by the script will be logged',type=str,default='time_taken.csv')

args = parser.parse_args()
print("args:", args)

# in a for loop, you need to create copy.deepcopy of params before proceeding

#for train_size in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]:
start_time = time.time()
#args.train_size_list = []
#args.shuffle_id_list = []
for train_size in args.train_size_list:
    for shuffle_id in args.shuffle_id_list:
    #for train_size in [100, 200, 400, 800, 1600, 3200,6400,12800,25600,51200]:
    #for train_size in [6400,12800,25600,51200]:
        print('sid:{}, train size: {}'.format(shuffle_id, train_size))
        #shuffle_id = 1
        #train_size = 310
        params = Params.from_file(args.params_file)
        settings.cuda = params['cuda_device'] != -1
        common_util.prepare_environment(params)
        params['train_size'] = train_size
        params['shuffle_id'] = shuffle_id
        params['train_data_path'] = os.path.join(
            params['data_dir'], 'shuffle'+str(shuffle_id), params['train_data_file'])
        params['validation_data_path'] = os.path.join(
            params['data_dir'], 'shuffle'+str(shuffle_id), params['validation_data_file'])
        
        
        
        semi_supervision = params.get('semi_supervised',False)
        which_mixer = params.get('which_mixer','cm')
        dd_warmup_iters = params.pop('dd_warmup_iters',1)
        dd_semi_warmup_iters = params.pop('dd_semi_warmup_iters',1)
        dd_update_freq = params.pop('dd_update_freq',2)
        constraints_wt = params.get('constraints_wt',0)
        calc_valid_freq = params.pop('calc_valid_freq', 1)
        backprop_after_xbatches = params.pop('backprop_after_xbatches',1)
        min_pct_of_unlabelled = params.pop('min_pct_of_unlabelled', 0.0)
        dd_increase_freq_after = params.pop('dd_increase_freq_after', None)
        dd_increase_freq_by = params.pop('dd_increase_freq_by', 1)
        dd_decay_lr = params.pop('dd_decay_lr',0)
        if semi_supervision:
            print("Semi Supervision On")
        
        #if semi_supervision:
        params['unlabelled_train_data_path'] = os.path.join(
            params['data_dir'], 'shuffle'+str(shuffle_id), params['unlabelled_train_data_file'])
        params['unlabelled_dataset_reader']['start_from'] = train_size
         
        params['dataset_reader']['how_many_sentences'] = train_size
        #params['model']['train_size'] = train_size
        params['serialization_dir'] = os.path.join(
            os.path.dirname(os.path.dirname(params['serialization_dir'])),
            'shuffle'+str(shuffle_id),
            'ts'+str(params['train_size']))

        serialization_dir = params['serialization_dir']
        training_util.create_serialization_dir(
            params, serialization_dir, args.recover, args.force)
        common_util.prepare_global_logging(serialization_dir, True)
        params.to_file(os.path.join(serialization_dir, CONFIG_NAME))
 
        for key in ['warmup_epochs','unlabelled_train_data_file','test_data_file', 'data_dir', 'cuda_device', 'serialization_dir', 'train_data_file', 'validation_data_file', 'constraints_wt', 'train_size', 'shuffle_id','semi_supervised','which_mixer','distributed_lambda_update']:
            params.pop(key,None)
        #Pdb().set_trace()
        pieces =gan_trainer_hm.TrainerPiecesForSemi.from_params(
            params, serialization_dir, args.recover, semi_supervision)  # pylint: disable=no-member

        trainer = Trainer.from_params(
            model=pieces.model,
            serialization_dir=serialization_dir,
            iterator=pieces.iterator,
            train_data=pieces.train_dataset,
            validation_data=pieces.validation_dataset,
            params=pieces.params,
            validation_iterator=pieces.validation_iterator)

        #pieces for constrained learning"
        constraints_model = Model.from_params(vocab=pieces.model.vocab, params=params.pop('dd_constraints'))
        dd_params = [[n, p] for n, p in constraints_model.named_parameters() if p.requires_grad]
        dd_optimizer = None
        if len(dd_params) > 0:
            dd_optimizer = Optimizer.from_params(dd_params, params.pop("dd_optimizer"))
        else:
            _ = params.pop('dd_optimizer')
        params.assert_empty('base train command')
        
        try:
            semi_trainer = gan_trainer_hm.SemiSupervisedTrainer(trainer, constraints_model, dd_optimizer, pieces.validation_iterator,  pieces.unlabelled_dataset, semi_supervision, which_mixer, dd_warmup_iters, dd_update_freq, constraints_wt, calc_valid_freq, backprop_after_xbatches, min_pct_of_unlabelled,dd_semi_warmup_iters, dd_increase_freq_after, dd_increase_freq_by,dd_decay_lr) 
                    
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

#args.shuffle_id_list = [1,2]
#args.train_size_list = [3,4]
end_time = time.time()
time_str = 'ts-{},sid-{},{},{}'.format('@'.join(map(str,args.train_size_list)), '@'.join(map(str,args.shuffle_id_list)),args.params_file,end_time-start_time)
fh = open(args.time_file,'a')
print(time_str,file=fh)
fh.close()



