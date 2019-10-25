from IPython.core.debugger import Pdb
import os
import sys

sys.path.append('../code')
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

#from allennlp.training.learning_rate_schedulers import LearningRateWithMetricsWrapper
from allennlp.predictors import SentenceTaggerPredictor

from torch.optim.lr_scheduler import ReduceLROnPlateau
from allennlp.nn import util as nn_util

from allennlp.common.params import Params
from allennlp.training import util as training_util
from allennlp.common import util as common_util
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
#from custom_trainer import CustomSrlTrainer, CustomSrlTrainerPieces
import argparse
import datetime
import logging
import shutil
import time
import re
import datetime
import traceback

from srl_custom_dataset import CustomSrlReader
from srl_custom_model import CustomSemanticRoleLabeler, write_to_conll_eval_file, write_bio_formatted_tags_to_file, write_conll_formatted_tags_to_file, convert_bio_tags_to_conll_format

from custom_span_based_f1_measure import CustomSpanBasedF1Measure
from allennlp.models.archival import archive_model, load_archive, CONFIG_NAME
from allennlp.data.iterators import BucketIterator
from allennlp.training.metrics import DecodeSpanBasedF1Measure

from helper import get_viterbi_pairwise_potentials, decode


parser = argparse.ArgumentParser()
parser.add_argument('--archive_dir', type=str)
parser.add_argument('--weights', type=str, default='best.th')
parser.add_argument('--pred_out', type=str,default='eval_custom_pred')
parser.add_argument('--gold_out', type=str,default='eval_custom_gold')
parser.add_argument('--batch_size', type=int,default=32)
parser.add_argument('--valid_data', action='store_true')
args = parser.parse_args()

#args.pred_out = os.path.join(args.archive_dir,args.pred_out)
#args.gold_out = os.path.join(args.archive_dir, args.gold_out)
args.weights = os.path.join(args.archive_dir,args.weights)



if not os.path.exists(os.path.dirname(args.pred_out)):
    os.makedirs(os.path.dirname(args.pred_out))

if not os.path.exists(os.path.dirname(args.gold_out)):
    os.makedirs(os.path.dirname(args.gold_out))


with torch.no_grad():
        config_file = os.path.join(args.archive_dir, "config.json")
        params = Params.from_file(config_file)

        archive = load_archive(args.archive_dir, cuda_device=0, weights_file=args.weights)

        model = archive.model
        params = archive.config 

        label_encoding="BIO"
        try:
                label_encoding = params["dataset_reader"]["label_encoding"]
        except:
                pass    

        print("Label encoding!", label_encoding)

        validation_iterator_params = params["iterator"]
        # validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=[("tokens", "num_tokens")])
        validation_iterator.index_with(model.vocab)

        dataset_name = 'test_data_path'
        if(args.valid_data):
                dataset_name = 'validation_data_path'

        print(f"Dataset: {dataset_name}")

        validation_and_test_dataset_reader = DatasetReader.from_params(params['dataset_reader'])
        validation_data_path = params[dataset_name]
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        val_generator = validation_iterator(validation_data, num_epochs=1, shuffle=False)       
        span_metric = CustomSpanBasedF1Measure(model.vocab, tag_namespace="labels", ignore_classes=["V"], label_encoding=label_encoding)
        decode_span_metric = DecodeSpanBasedF1Measure(model.vocab, tag_namespace="labels", ignore_classes=["V"], label_encoding=label_encoding)

        model.eval()

        print("Done")

        # index to token vocabulary
        index2label = model.vocab.get_index_to_token_vocabulary(namespace="labels")

        prediction_file = open(f"{args.pred_out}.txt", "w")
        gold_file = open(f"{args.gold_out}.txt", "w")

        prediction_file_INFDEC = open(f"{args.pred_out}_INFDEC.txt", "w")
        gold_file_INFDEC = open(f"{args.gold_out}_INFDEC.txt", "w")     

        vocab = model.vocab
        
        transition_matrix = get_viterbi_pairwise_potentials(vocab, label_encoding)      

        print("Starting batch with batch size 1")
        for (i, batch) in enumerate(val_generator):
                if(i % 10 == 0):
                        print(f"Batch {i}")
                # print("Moving batch to GPU")
                batch = nn_util.move_to_device(batch,0)
                scores = model(
                                                tokens=batch["tokens"], 
                                                verb_indicator=batch["verb_indicator"], 
                                                tags=batch["tags"], 
                                                metadata=batch["metadata"]
                                          )
                
                pred_probs = scores["class_probabilities"]
                argmax_pred_probs = pred_probs.max(-1)[1]
                
                all_pred_labels = decode(transition_matrix, vocab, output_dict=scores)["tags"]

                bsize = pred_probs.shape[0]
                senlen = argmax_pred_probs.shape[1]

                for j in range(bsize):
                        num_words = int(scores['mask'][j].sum())
                        sentence = batch["metadata"][j]["words"][:num_words]
                        gold_labels = batch["metadata"][j]["gold_tags"][:num_words]
                        verb_tensor = batch["verb_indicator"][j]
                        verb_indicator = int(verb_tensor.argmax())
                        
                        pred_labels = []
                        # Viterbi decoding
                        pred_labels_INFDEC = all_pred_labels[j][:num_words]
                        # Regular decoding
                        for k in range(senlen):
                                pred_labels.append(index2label[int(argmax_pred_probs[j, k])])
                        pred_labels = pred_labels[:num_words]
                                

                        # print(sentence)
                        # print(gold_labels)            
                        # print(pred_labels)
                        # print(decoded_tags)
                        # print("\n")

                        write_bio_formatted_tags_to_file(prediction_file,
                                                                                         gold_file,
                                                                                         verb_indicator,
                                                                                         sentence,
                                                                                         pred_labels,
                                                                                         gold_labels)

                        write_bio_formatted_tags_to_file(prediction_file_INFDEC,
                                                                                         gold_file_INFDEC,
                                                                                         verb_indicator,
                                                                                         sentence,
                                                                                         pred_labels_INFDEC,
                                                                                         gold_labels)

                        # convert to strings!


        prediction_file.close()
        prediction_file_INFDEC.close()
        gold_file.close()
        gold_file_INFDEC.close()
        
