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

from allennlp.predictors import SentenceTaggerPredictor

from torch.optim.lr_scheduler import ReduceLROnPlateau
from allennlp.nn import util as nn_util

from allennlp.common.params import Params
from allennlp.training import util as training_util
from allennlp.common import util as common_util
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
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
from penalty import SpanImplicationMax, Transition, TerminalTag
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode

from custom_span_based_f1_measure import CustomSpanBasedF1Measure
from allennlp.models.archival import archive_model, load_archive, CONFIG_NAME
from allennlp.data.iterators import BucketIterator
from allennlp.training.metrics import DecodeSpanBasedF1Measure


from helper import get_viterbi_pairwise_potentials, decode, count_violating_spans, get_span_list 
import settings

settings.cuda = True

parser = argparse.ArgumentParser()
parser.add_argument('--archive_dir', type=str)
parser.add_argument('--weights', type=str,default='best.th')
parser.add_argument('--valid_data', action='store_true')
parser.add_argument('--use_maxp', type=str, default="false")
parser.add_argument('--max_penalty', type=str, default="negate")
parser.add_argument('--aggregate_type', type=str, default="max")
parser.add_argument('--out_file', type=str,default='get_violations')
parser.add_argument('--batch_size', type=int,default=32)

args = parser.parse_args()

#args.out_file = os.path.join(args.archive_dir, args.out_file)

with torch.no_grad():
    config_file = os.path.join(args.archive_dir, "config.json")
    params = Params.from_file(config_file)

    archive = load_archive(args.archive_dir, cuda_device=0, weights_file=os.path.join(args.archive_dir, args.weights))

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
    validation_iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=[("tokens", "num_tokens")], biggest_batch_first = True)
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

    # vocab = model.vocab
    
    transition_matrix = get_viterbi_pairwise_potentials(model.vocab, label_encoding)        

    if args.use_maxp.lower() not in ["true", "false"]:
        exit()

    if args.use_maxp.lower() == "true":
        my_use_maxp = True
    else:
        my_use_maxp = False

    config_span_sum = {"weight": 1, "use_maxp": False, "aggregate_type": "sum", "max_penalty": "negate"}
    config_span_max = {"weight": 1, "use_maxp": False, "aggregate_type": "max", "max_penalty": "negate"}
    config_terminal = {"weight": 1}
    config_constr_trans_sum = {"weight": 1, "aggregate_type": "sum"}
    config_constr_trans_max = {"weight": 1, "aggregate_type": "max"}
    
    constraint_dict_span_sum = SpanImplicationMax(model.vocab, config_span_sum, "BL")
    constraint_dict_span_max = SpanImplicationMax(model.vocab, config_span_max, "BL")
    constraint_dict_Bterminal = TerminalTag(model.vocab, config_terminal, "B")
    constraint_dict_Lterminal = TerminalTag(model.vocab, config_terminal, "L")
    constraint_dict_span_sum_LB = SpanImplicationMax(model.vocab, config_span_sum, "LB")
    constraint_dict_trans_sum = Transition(model.vocab, config_constr_trans_sum)
    constraint_dict_trans_max = Transition(model.vocab, config_constr_trans_max)
    if label_encoding != "BIOUL":
        constraint_dicts = {"trans_sum": constraint_dict_trans_sum}
    else:
        constraint_dicts = {
                      "span_sum": constraint_dict_span_sum,
                      "span_max": constraint_dict_span_max,
                      "span_sum_LB": constraint_dict_span_sum_LB,
                      "trans_sum": constraint_dict_trans_sum,
                      "trans_max": constraint_dict_trans_max,
                      "Bterminal": constraint_dict_Bterminal,
                      "Lterminal": constraint_dict_Lterminal
                    }


    tags_out_nondecode = open(f"{args.out_file}_tags_nondecode.txt", "w")
    tags_out_decode = open(f"{args.out_file}_tags_decode.txt", "w")
    tags_gold = open(f"{args.out_file}_gold.txt", "w")

    list_keys = ["span_max", "span_sum", "span_sum_LB", "trans_max", "trans_sum", "Bterminal", "Lterminal"]

    total_nondecode_penalty = {key:0 for key in list_keys + ['num_violating_spans']}
    total_decode_penalty = {key:0 for key in list_keys+ ['num_violating_spans']}

    nondecode_sentence_avg_disagreement_rate = {key:0 for key in list_keys+['num_violating_spans']}
    decode_sentence_avg_disagreement_rate = {key:0 for key in list_keys+['num_violating_spans']}

    num_semantic_spans_decode = 0
    num_semantic_spans_decode_nonid = 0
    num_semantic_spans_nondecode = 0
    num_semantic_spans_nondecode_nonid = 0
    
    num_sentences = 0       

    total_num_words = 0
    total_gold_spans = 0
    total_gold_spans_nonid = 0

    print("Starting batch with batch size 1")
    for (i, batch) in enumerate(val_generator):
        #Pdb().set_trace()
        syntax_span_matrix = batch['span_matrix'].cpu()
        if(i % 100 == 0):
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
        
        bsize = pred_probs.shape[0]
        senlen = argmax_pred_probs.shape[1]

        out_dict = decode(transition_matrix, model.vocab, output_dict=scores)
        decoded_pred_labels = out_dict["tags"]
        all_seq_prob = out_dict["max_seq"]

        decode_prob = []
        sequence_lengths = get_lengths_from_binary_sequence_mask(out_dict["mask"]).data.tolist()
        # Pdb().set_trace()
        max_seq_length = max(sequence_lengths)
        for j in range(bsize):
            pseq = ['O' for _ in range(max_seq_length)]
            pseq[:len(decoded_pred_labels[j])] = decoded_pred_labels[j]
            pseq = [model.vocab.get_token_to_index_vocabulary(namespace="labels")[key] for key in pseq]
            decode_prob.append(pseq)                        


        decode_prob = torch.LongTensor(decode_prob)
        decode_prob = decode_prob.cuda()

        ohot_pred_probs = torch.zeros_like(pred_probs)
        ohot_decode_prob = torch.zeros_like(pred_probs)

        pred_indices = pred_probs.max(dim=2)[1]
        
        ohot_pred_probs = ohot_pred_probs.scatter(2, index=pred_indices.unsqueeze(-1), source=1.0)
        ohot_decode_prob = ohot_decode_prob.scatter(2, index=decode_prob.unsqueeze(-1), source=1.0)

        # for i in range(pred_probs.shape[0]):
        #       for j in range(pred_probs.shape[1]):
        #               ohot_pred_probs[i,j,pred_indices[i,j]] = 1
        #               ohot_decode_prob[i,j,decode_prob[i,j]] = 1
        # ohot_pred_probs[] = 1
        # ohot_decode_prob[decode_prob] = 1

        this_batch_nondecode_penalty = {key:0 for key in list_keys}
        this_batch_decode_penalty = {key:0 for key in list_keys}
        nondecode_sentence_wise_penalty = {}
        decode_sentence_wise_penalty = {}

        for key in list_keys:
            this_batch_nondecode_penalty[key],_ = constraint_dicts[key].get_penalty(ohot_pred_probs, scores["mask"].float(), batch["span_matrix"])
            this_batch_decode_penalty[key],_ = constraint_dicts[key].get_penalty(ohot_decode_prob, scores["mask"].float(), batch["span_matrix"])
            total_nondecode_penalty[key]  += this_batch_nondecode_penalty[key].sum().item()
            total_decode_penalty[key]  += this_batch_decode_penalty[key].sum().item()               

            nondecode_sentence_wise_penalty[key] = this_batch_nondecode_penalty[key].sum(dim=1)
            decode_sentence_wise_penalty[key] = this_batch_decode_penalty[key].sum(dim=1)           

        for j in range(bsize):
            num_sentences += 1
            num_words = int(scores['mask'][j].sum())
            total_num_words += num_words+1
            gold_labels = batch["metadata"][j]["gold_tags"][:num_words]
            pred_labels_INFDEC = decoded_pred_labels[j][:num_words]
            pred_labels = []
            for k in range(senlen):
                pred_labels.append(index2label[int(argmax_pred_probs[j, k])])
            pred_labels = pred_labels[:num_words]

            this_sentence_nondecode_penalty = {}
            this_sentence_decode_penalty = {}

            for key in list_keys:
                this_sentence_nondecode_penalty[key] = float(nondecode_sentence_wise_penalty[key][j])
                this_sentence_decode_penalty[key] = float(decode_sentence_wise_penalty[key][j])
            
            tags_out_nondecode.write(" ".join(pred_labels)+'\n '+" ".join([str(this_sentence_nondecode_penalty[key]) for key in list_keys])+'\n')
            tags_out_decode.write(" ".join(pred_labels_INFDEC)+'\n '+" ".join([str(this_sentence_decode_penalty[key]) for key in list_keys])+'\n')
            tags_gold.write(" ".join(gold_labels)+'\n\n')

            gold_span_list, num_gold_spans = get_span_list(gold_labels) 
            total_gold_spans += num_gold_spans  
            total_gold_spans_nonid += len(gold_span_list)  
            
            decode_this_sentence_total_spans = 0
            nondecode_this_sentence_total_spans = 0


            decode_semantic_span_list, num_spans = get_span_list(pred_labels_INFDEC)
            num_semantic_spans_decode += num_spans
            num_semantic_spans_decode_nonid += len(decode_semantic_span_list)
            decode_this_sentence_total_spans += num_spans 
            #count violating spans
            decode_num_violating_spans_this_sentence = count_violating_spans(decode_semantic_span_list, syntax_span_matrix[j])

            nondecode_semantic_span_list, num_spans = get_span_list(pred_labels)            
            num_semantic_spans_nondecode += num_spans
            num_semantic_spans_nondecode_nonid += len(nondecode_semantic_span_list)
            nondecode_this_sentence_total_spans += num_spans 
            nondecode_num_violating_spans_this_sentence = count_violating_spans(nondecode_semantic_span_list, syntax_span_matrix[j])
            

            total_decode_penalty['num_violating_spans'] += decode_num_violating_spans_this_sentence
            total_nondecode_penalty['num_violating_spans'] += nondecode_num_violating_spans_this_sentence 
            
            this_sentence_decode_penalty['num_violating_spans'] =decode_num_violating_spans_this_sentence
            this_sentence_nondecode_penalty['num_violating_spans'] = nondecode_num_violating_spans_this_sentence

            if decode_this_sentence_total_spans == 0:
                decode_this_sentence_avg_disagreement_rate = {key:0 for key in this_sentence_decode_penalty.keys()}
            else:
                decode_this_sentence_avg_disagreement_rate = {}
                for key in this_sentence_decode_penalty.keys():
                    decode_this_sentence_avg_disagreement_rate[key] = (1.0*this_sentence_decode_penalty[key]/decode_this_sentence_total_spans)

            if nondecode_this_sentence_total_spans == 0:
                nondecode_this_sentence_avg_disagreement_rate = {key:0 for key in this_sentence_nondecode_penalty.keys()}
            else:
                nondecode_this_sentence_avg_disagreement_rate = {}
                for key in this_sentence_nondecode_penalty.keys():
                    nondecode_this_sentence_avg_disagreement_rate[key] = (1.0*this_sentence_nondecode_penalty[key]/nondecode_this_sentence_total_spans)

            for key in this_sentence_decode_penalty.keys():
                decode_sentence_avg_disagreement_rate[key] += decode_this_sentence_avg_disagreement_rate[key]
                nondecode_sentence_avg_disagreement_rate[key] += nondecode_this_sentence_avg_disagreement_rate[key]

        # [','.join(list(map(str,x))) for x in this_batch_nondecode_penalty.cpu().numpy()]

    
    nondecode_res_out = open(f"{args.out_file}_result_nondecode", "w")
    decode_res_out = open(f"{args.out_file}_result_decode", "w")
    
    nondecode_res_out.write(f"Number of total word transitions = {total_num_words}\n")
    nondecode_res_out.write("Number of constraint violations:\n")
    for key in total_nondecode_penalty.keys():
        nondecode_res_out.write(f"{key}: {total_nondecode_penalty[key]}\n")

    nondecode_BL_B_total = total_nondecode_penalty["span_sum"]+total_nondecode_penalty["Bterminal"]
    nondecode_LB_L_total = total_nondecode_penalty["span_sum_LB"]+total_nondecode_penalty["Lterminal"]
    nondecode_res_out.write(f"BL+B violations = {nondecode_BL_B_total}\n")
    nondecode_res_out.write(f"LB+L violations = {nondecode_LB_L_total}\n")

    nondecode_res_out.write(f"Total semantic spans: {num_semantic_spans_nondecode}\n")
    nondecode_res_out.write(f"Total semantic spans non id: {num_semantic_spans_nondecode_nonid}\n")
    nondecode_res_out.write(f"Total gold semantic spans: {total_gold_spans}\n")
    nondecode_res_out.write(f"Total gold semantic spans non id: {total_gold_spans_nonid}\n")
    for key in nondecode_sentence_avg_disagreement_rate.keys():
        nondecode_res_out.write(f"Total disagreement rate {key}: {(1.0*total_nondecode_penalty[key])/num_semantic_spans_nondecode}\n")
        nondecode_res_out.write(f"Average disagreement rate {key}: {1.0*nondecode_sentence_avg_disagreement_rate[key]/num_sentences}\n")

    decode_res_out.write(f"Number of total word transitions = {total_num_words}\n")
    decode_res_out.write("Number of constraint violations:\n")
    for key in total_decode_penalty.keys():
        decode_res_out.write(f"{key}: {total_decode_penalty[key]}\n")

    decode_BL_B_total = total_decode_penalty["span_sum"]+total_decode_penalty["Bterminal"]
    decode_LB_L_total = total_decode_penalty["span_sum_LB"]+total_decode_penalty["Lterminal"]
    decode_res_out.write(f"BL+B violations = {decode_BL_B_total}\n")
    decode_res_out.write(f"LB+L violations = {decode_LB_L_total}\n")

    decode_res_out.write(f"Total semantic spans: {num_semantic_spans_decode}\n")
    decode_res_out.write(f"Total semantic spans non id: {num_semantic_spans_decode_nonid}\n")
    decode_res_out.write(f"Total gold semantic spans: {total_gold_spans}\n")
    decode_res_out.write(f"Total gold semantic spans non id: {total_gold_spans_nonid}\n")

    for key in decode_sentence_avg_disagreement_rate.keys():
        decode_res_out.write(f"Total disagreement rate {key}: {(1.0*total_decode_penalty[key])/num_semantic_spans_decode}\n")
        decode_res_out.write(f"Average disagreement rate {key}: {1.0*decode_sentence_avg_disagreement_rate[key]/num_sentences}\n")

    decode_res_out.close()
    nondecode_res_out.close()

