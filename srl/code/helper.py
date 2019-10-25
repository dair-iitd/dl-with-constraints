from IPython.core.debugger import Pdb
from typing import Dict, List, TextIO, Optional, Any
import warnings

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.training.metrics import DecodeSpanBasedF1Measure
from allennlp.modules.conditional_random_field import allowed_transitions

def get_viterbi_pairwise_potentials(vocab, label_encoding):
    """
    Generate a matrix of pairwise transition potentials for the BIO labels.
    The only constraint implemented here is that I-XXX labels must be preceded
    by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
    constraint, pairs of labels which do not satisfy this constraint have a
    pairwise potential of -inf.

    Returns
    -------
    transition_matrix : torch.Tensor
        A (num_labels, num_labels) matrix of pairwise potentials.
    """
    all_labels = vocab.get_index_to_token_vocabulary("labels")
    num_labels = len(all_labels)
    transition_matrix = torch.zeros([num_labels+2, num_labels+2]).fill_(float("-inf"))

    constraints = allowed_transitions(label_encoding, all_labels)
    # print(constraints) 
    for c in constraints:
        #if (c[0] < num_labels) and (c[1] < num_labels):
        transition_matrix[c[0],c[1]] = 0.0
    #
    return transition_matrix
    """
    for i, previous_label in all_labels.items():
        for j, label in all_labels.items():
            # I labels can only be preceded by themselves or
            # their corresponding B tag.
            if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                transition_matrix[i, j] = float("-inf")
    
    return transition_matrix
    """

def decode(transition_matrix, vocab, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
    constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
    ``"tags"`` key to the dictionary with the result.
    """
    all_predictions = output_dict['class_probabilities']
    sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()
    max_seq_length = max(sequence_lengths)
    if all_predictions.dim() == 3:
        predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
    else:
        predictions_list = [all_predictions]
    all_tags = []    
    all_prob_seq = []
    num_tags  = transition_matrix.shape[0] - 2 
    tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)
    for predictions, length in zip(predictions_list, sequence_lengths):
        tag_sequence.fill_(-10000.)
        tag_sequence[0, num_tags] = 0. 
        tag_sequence[1:(length + 1), :num_tags] = predictions[:length]
        tag_sequence[(length+1),num_tags+1] = 0.

        max_likelihood_sequence, _  = viterbi_decode(tag_sequence[:(length+2)], transition_matrix)
        max_likelihood_sequence = max_likelihood_sequence[1:-1]
        tags = [vocab.get_token_from_index(x, namespace="labels")
                for x in max_likelihood_sequence]
        all_tags.append(tags)
        all_prob_seq.append(max_likelihood_sequence)
    output_dict['tags'] = all_tags
    output_dict['max_seq'] = all_prob_seq
    return output_dict



def count_violating_spans(decode_semantic_span_list, syntax_span_matrix):
    num_violations = 0
    for this_span in decode_semantic_span_list:
        start_idx = this_span[0]
        end_idx = this_span[1]
        assert (start_idx <= end_idx)
        assert (end_idx <= syntax_span_matrix.shape[0])
        if syntax_span_matrix[start_idx,end_idx] == 0:
            num_violations += 1
    #
    return num_violations


def get_span_list(pred_labels_INFDEC):
    # pred_labels_INFDEC
    cur_tag_type = 'O'
    span_active = False
    decode_semantic_span_list = []
    total_spans = 0
    for this_ind,label in enumerate(pred_labels_INFDEC):
        if(label == 'O'):
            lb_posn = label
            lb_type = label
        else:
            lb_posn = label[0]
            lb_type = label[2:]
       
        if span_active:
            if lb_posn == 'B':
                cur_tag_type = lb_type
                current_span = [this_ind]
            elif lb_posn == 'I':
                if(cur_tag_type != lb_type):
                    span_active = False
            elif lb_posn == 'U':
                span_active = False
                # QUESTION - increase semantic span count?
                total_spans += 1
            elif lb_posn == 'L':
                span_active = False
                if(cur_tag_type == lb_type):
                    total_spans += 1
                    current_span.append(this_ind)
                    decode_semantic_span_list.append(current_span)
            else:
                span_active = False
        else:
            if lb_posn == 'B':
                span_active = True
                current_span = [this_ind]
                cur_tag_type = lb_type
            elif lb_posn == 'U':
                total_spans += 1
            else: # I-, L-, O
                continue
    return decode_semantic_span_list, total_spans 

"""
def write_output_to_file(batch, output_dict, ind2lab, penalties, global_count, fh=None):
    #Pdb().set_trace()
    topk,topk_ind = output_dict['class_probabilities'].detach().topk(k = 5, dim=-1)
    topk = topk.detach().cpu().numpy()
    topk_ind = topk_ind.detach().cpu().numpy()
    all_sentences = batch['metadata']
    vio_keys = [x for x in penalties.keys() if 'violations' in x]
    con_keys = [x.split('_')[0] for x in vio_keys]
    sum_violations = None
    batch_size,num_words = batch['tokens']['tokens'].shape
    for k in vio_keys: 
        if sum_violations is None:
            sum_violations = penalties[k][:,:num_words].detach()
        else:
            sum_violations += penalties[k][:,:num_words].detach()
    #
    sum_violations = sum_violations*(output_dict['mask'].detach().float())
    for i,this_sentence in enumerate(all_sentences):
        #check if anything wrong or not
        flag = sum_violations[i].sum() > 0
        tot_incorrect = 0
        for j in range(len(this_sentence['words'])):
            gold_tag = this_sentence['gold_tags'][j]
            pred_tag = ind2lab[topk_ind[i,j,0]]
            if pred_tag != gold_tag:
                tot_incorrect += 1
            
        if tot_incorrect > 0:
            print('#gc: {} i {} VERB=={}. Total pen: {}. #Incorrect: {}'.format(global_count, i,this_sentence['verb'], round(sum_violations[i].sum().item(),3), tot_incorrect),file=fh)
            for j in range(len(this_sentence['words'])):
                word = this_sentence['words'][j]
                gold_tag = this_sentence['gold_tags'][j]
                top5_pred = ','.join(list(map(lambda x: ind2lab[x], topk_ind[i,j])))
                top5_prob = ','.join(list(map(lambda x: str(round(x,4)), topk[i,j])))
                pen = ','.join([con_key+"_"+str(round(penalties[vio_key][i,j].item(),4)) for vio_key,con_key in zip(vio_keys,con_keys)])
                print('{} ; {} ; {} ; {} ; {}'.format(word,gold_tag,top5_pred,top5_prob, pen),file=fh)
        #
        global_count += 1
    return global_count
"""

def write_decoded_output_to_file(batch, output_dict, ind2lab, penalties, global_count, decoded_output=None, fh=None):
    #Pdb().set_trace()
    topk,topk_ind = output_dict['class_probabilities'].detach().topk(k = 5, dim=-1)
    topk = topk.detach().cpu().numpy()
    topk_ind = topk_ind.detach().cpu().numpy()
    all_sentences = batch['metadata']
    vio_keys = [x for x in penalties.keys() if 'violations' in x]
    con_keys = [x.split('_')[0] for x in vio_keys]
    sum_violations = None
    batch_size,num_words = batch['tokens']['tokens'].shape
    for k in vio_keys: 
        if sum_violations is None:
            sum_violations = penalties[k][:,:num_words].clone().detach()
        else:
            sum_violations += penalties[k][:,:num_words].detach()
    #
    sum_violations = sum_violations*(output_dict['mask'].detach().float())
    for i,this_sentence in enumerate(all_sentences):
        #check if anything wrong or not
        flag = sum_violations[i].sum() > 0
        tot_incorrect = 0
        for j in range(len(this_sentence['words'])):
            gold_tag = this_sentence['gold_tags'][j]
            pred_tag = ind2lab[topk_ind[i,j,0]]
            if pred_tag != gold_tag:
                tot_incorrect += 1
            
        if tot_incorrect > 0:
            tot_pen = ','.join([con_key+"_"+str(round(penalties[vio_key][i].sum().item(),4)) for vio_key,con_key in zip(vio_keys,con_keys)])

            print('#gc: {} i {} VERB=={}. #Words: {}  #Incorrect: {}. Tot. Penalty: {} Penalty: {}'.format(global_count, i,this_sentence['verb'], len(this_sentence['words']), tot_incorrect, round(sum_violations[i].sum().item(),4), tot_pen ),file=fh)
            for j in range(len(this_sentence['words'])):
                word = this_sentence['words'][j]
                gold_tag = this_sentence['gold_tags'][j]
                top5_pred = ','.join(list(map(lambda x: ind2lab[x], topk_ind[i,j])))
                top5_prob = ','.join(list(map(lambda x: str(round(x,4)), topk[i,j])))
                pen = ','.join([con_key+"_"+str(round(penalties[vio_key][i,j].item(),4)) for vio_key,con_key in zip(vio_keys,con_keys)])
                do = 'NA'
                if decoded_output is not None:
                    do = decoded_output[i][j]

                print('{} ; {} ; {} ; {} ; {} ; {}'.format(word,gold_tag, do, top5_pred,top5_prob, pen),file=fh)
        #
        global_count += 1
    return global_count


