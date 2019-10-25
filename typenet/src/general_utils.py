from __future__ import print_function
import math
import time
import os, sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
from data_iterator import MAX_SENT
from itertools import izip

def split_title_line(title_text, split_on='(', max_words=5):  # , max_words=None):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    split_at = title_text.find (split_on)
    ti = title_text
    if split_at > 1:
        ti = ti.split (split_on)
        for i, tx in enumerate (ti[1:]):
            ti[i + 1] = split_on + tx
    if type (ti) == type ('text'):
        ti = [ti]
    for j, td in enumerate (ti):
        if td.find (split_on) > 0:
            pass
        else:
            tw = td.split ()
            t2 = []
            for i in range (0, len (tw), max_words):
                t2.append (' '.join (tw[i:max_words + i]))
            ti[j] = t2
    ti = [item for sublist in ti for item in sublist]
    ret_tex = []
    for j in range (len (ti)):
        for i in range(0, len(ti)-1, 2):
            if len (ti[i].split()) + len (ti[i+1].split ()) <= max_words:
                mrg = " ".join ([ti[i], ti[i+1]])
                ti = [mrg] + ti[2:]
                break

    if len (ti[-2].split ()) + len (ti[-1].split ()) <= max_words:
        mrg = " ".join ([ti[-2], ti[-1]])
        ti = ti[:-2] + [mrg]
    return '\n'.join (ti)

def compute_param_norm(parameters, norm_type= 2): 
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0 
    for p in parameters:
        if p.is_sparse:
            # need to coalesce the repeated indices before finding norm
            grad = p.data.coalesce()
            param_norm = grad._values().norm(norm_type)
        else:
            param_norm = p.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def compute_grad_norm(parameters, norm_type= 2): 
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0 
    for p in parameters:
        if p.grad.is_sparse:
            # need to coalesce the repeated indices before finding norm
            grad = p.grad.data.coalesce()
            param_norm = grad._values().norm(norm_type)
        else:
            param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm 


#used in logging
def add_prefix(d):
    if 'prefix' in d:
        p = d.pop('prefix')
    else:
        p = ''
    return dict([(p+str(k),v) for k,v in d.items()])


# get weight imbalance

def get_loss_weights(y,config):
    if config.class_imb == 0:
        wts = torch.ones_like(y)
    elif config.class_imb == -1:
        pc = y.sum(dim=1) + 1.0
        nc = y.shape[1] - pc + 1.0
        wts = (y*((nc/pc).unsqueeze(-1)) + (1-y))
    elif config.class_imb == 1:
        pc = y.sum().item()  + 1.0
        nc = y.numel() - pc + 1.0
        wts = (y*(nc/pc) + (1-y))
    else:
        wts =  (y*(config.class_imb) + (1-y))
    
    return (wts*(wts.numel()/wts.sum()))


# Functions to create data mixers - supervised mix unsupervised


####
"""
for every batch from g2, generate 'ratio' batches from g1. Assuming g1 is bigger. 
"""
def _mix_generators(g1,g2,g1_size,g2_size, g1_id =0):
    count = 0
    g1_active = True
    g2_active = True  
    g2_id = (g1_id + 1) %2 
    ratio  = int(round(g1_size/g2_size)) + 1
    #for every batch from g2, generate 'ratio' batches from g1
    while True:
        if (g2_active and (count % ratio == 0)):
            try:
                yield (g2.next(),g2_id)
            except StopIteration:
                g2_active = False
        elif g1_active:
            try:
                yield (g1.next(),g1_id)
            except StopIteration:
                g1_active = False
        #
        count += 1
        if not (g1_active or g2_active):
            break
    #


def mix_generators_bm(g1,g2,g1_size, g2_size,g1_id = 0):
    if g1_size < g2_size:
        return _mix_generators(g2,g1,g2_size,g1_size,(g1_id+1)%2)
    else:
        return _mix_generators(g1,g2,g1_size,g2_size,g1_id)

def mix_generators_cm(g1, g2, id1):
    for (x1,x2) in izip(g1,g2):
        yield (x1,id1)
        yield (x2, (id1 + 1)%2)

def mix_generators_em(g1,g2,id1):
    for x in g1:
        yield(x,id1)
    for x in g2:
        yield(x,(id1 + 1)%2)


def get_mixer(ds1, ds2, id1 = 0, which_mixer='bm'):
    #Pdb().set_trace()
    if ds2 is None:
        num_batches1 = math.ceil(len(ds1)/ds1.batch_size)
        g1 = ds1.get_batch()
        return (mix_generators_em(g1,[],id1),num_batches1)

    num_batches1 = math.ceil(len(ds1)/ds1.batch_size)
    num_batches2 = math.ceil(len(ds2)/ds2.batch_size)
    id2 = (id1+1)%2
    if (which_mixer in ['em','bm']) or (num_batches1 == 0) or (num_batches2 == 0):
        total_batches = num_batches1+num_batches2
        num_epochs1 = 1
        num_epochs2 = 1
    elif which_mixer == 'cm':
        total_batches = 2*max(num_batches1, num_batches2)
        num_epochs1 = math.ceil(total_batches/(2*num_batches1))
        num_epochs2 = math.ceil(total_batches/(2*num_batches2))
    else:
        raise "incorrect value of which_mixer {}".format(which_mixer)
    #raw_g1 = it1(ds1, num_epochs = num_epochs1)
    #raw_g2 = it2(ds2, num_epochs = num_epochs2)
    g1 = ds1.get_batch(num_epochs = int(num_epochs1))
    g2 = ds2.get_batch(num_epochs = int(num_epochs2))
    if (which_mixer == 'em') or (num_batches1 == 0) or (num_batches2 == 0):
        mixer = mix_generators_em(g1,g2,id1)
    elif which_mixer == 'bm':
        mixer = mix_generators_bm(g1,g2,num_batches1, num_batches2,id1)
    elif which_mixer == 'cm':
        mixer = mix_generators_cm(g1, g2,id1)
    #
    return (mixer, total_batches)
 


# ===== Some utility for padding and creating torch index tensors =====

def _pad_sequences(sequences, pad_tok, max_length, pos):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        if pos:
            seq_ = seq + [idx for idx in xrange(seq[-1]+1, seq[-1] + max_length - len(seq)+1)]
        else:
            seq_ = seq + [pad_tok]*max(max_length - len(seq), 0)


        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return np.array(sequence_padded), np.array(sequence_length)

def pad_sequences(sequences, pad_tok, torchify_op, on_gpu, pos = False):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        Padded and torchified tensors (along with true lengths also torchified) and transferred to GPU
    """
    max_id, max_length_seq = max(enumerate(sequences), key = lambda id : len(id[1]))
    max_length = len(max_length_seq)
    sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length, pos)

    if on_gpu:
        return Variable(torchify_op(sequence_padded).cuda() ), Variable(torchify_op(sequence_length).cuda())
    else:
        return Variable(torchify_op(sequence_padded) ), Variable(torchify_op(sequence_length))

def pad2d(sequences, pad_tok, torchify_op, on_gpu):
    '''

    :param sequences: A list of list of list where the 2nd dimension is for word characters and first dim
                                is for words
    :param pad_tok: everything will be padded with this (usually PAD)
    :param torchify_op:
    :param on_gpu:
    :return: padded, and good for inputting into nn.Module
    '''


    max_length_word = max([max([len(x) for x in seq]) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, pad_tok, max_length_word, False)
        sequence_padded += [sp]
        sequence_length += [sl-1]

    max_length_sentence = max([len(x) for x in sequences])
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word,
                                        max_length_sentence, False)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence, False)

    if on_gpu:
        return Variable(torchify_op(sequence_padded).cuda()), Variable(torchify_op(sequence_length).cuda())
    else:
        return Variable(torchify_op(sequence_padded)), Variable(torchify_op(sequence_length))






# === Functions for loading and saving model ====
def save_checkpoint(state, epoch, is_best, config):
    checkpoint_file = os.path.join(config.checkpoint_file, config.model_file)
    best_file  = checkpoint_file+'_best'
    if is_best:
        checkpoint_file  = best_file 

    torch.save(state, checkpoint_file)
    logging.info('Saving checkpoint to {}'.format(checkpoint_file))


def generate_checkpoint(config,model,constraints,optim1,optim2,epoch,num_iter,best_score,is_best,num_iter_total,lambda_iter, last_lambda_update, dd_update_freq):
    cpoint = {	
            'epoch': epoch,
            'best_score': best_score,
            'model': model.state_dict(),
            'constraints': constraints.state_dict(),
            'optim1': optim1.state_dict() if optim1 is not None else None, 
            'optim2': optim2.state_dict() if optim2 is not None else None,
            'is_best': is_best,
            'num_iter': num_iter,
            'num_iter_total': num_iter_total,
            'lambda_iter': lambda_iter,
            'last_lambda_update': last_lambda_update,
            'dd_update_freq': dd_update_freq 
            }
    return cpoint

def load_checkpoint(config,model=None,constraints=None,optim1=None,optim2 = None,best=False):
    checkpoint_file = os.path.join(config.checkpoint_file,config.model_file)
    best_file = checkpoint_file + '_best'
    if best:
        print("Loading from best ")
        checkpoint_file = best_file
    start_epoch = 0
    num_iter = 0
    num_iter_total = 0
    lambda_iter = 0
    best_score  = -9999999
    last_lambda_update = 0
    dd_update_freq = config.dd_update_freq 
    if os.path.exists(checkpoint_file):
        logging.info('Starting from checkpoint: {}'.format(checkpoint_file))
        cp = torch.load(checkpoint_file)
        start_epoch = cp['epoch'] + 1
        num_iter = cp.get('num_iter',0) 
        num_iter_total = cp.get('num_iter_total',0)
        lambda_iter = cp.get('lambda_iter',0)
        last_lambda_update = cp.get('last_lambda_update',0) 
        dd_update_freq = cp.get('dd_update_freq', config.dd_update_freq) 
        if model is not None:
            model.load_state_dict(cp['model'])
        if constraints is not None:
            constraints.load_state_dict(cp['constraints'])
        if optim1 is not None:
            optim1.load_state_dict(cp['optim1'])
        if optim2 is not None:
            optim2.load_state_dict(cp['optim2'])

        best_score = cp['best_score']
        del cp
    else:
        logging.info("Chheckpoint file not found:{}".format(checkpoint_file))
    return start_epoch, best_score, num_iter, num_iter_total, lambda_iter,last_lambda_update, dd_update_freq

def save_model(model, config):
    filename = "%s/%s.checkpoint" %(config.checkpoint_file, config.model_file)

    try:
        torch.save(model.state_dict(), filename)
    except BaseException:
        pass


def load_model(model, config):
    fname = "%s/%s.checkpoint" %(config.checkpoint_file, config.model_file)
    if os.path.isfile(fname):
        print("Loading model from {}".format(fname))
        model.load_state_dict(torch.load(fname))

#copied from allennlp.trainer.util
def sparse_clip_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad.is_sparse:
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad.is_sparse:
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


# === performing gradient descent
#copied from allennlp.trainer.util
def rescale_gradients(model, grad_norm = None):
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    """
    if grad_norm:
        parameters_to_clip = [p for p in model.parameters()
                              if p.grad is not None]
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None


def step(model, loss, optimizer, clip_val = None,grad_norm = None):
    optimizer.zero_grad()
    loss.backward()
    # clip gradients
    #grad norm before clipping
    parameters = [p for p in model.parameters() if p.grad is not None]
    grad_norm_before_clip = compute_grad_norm(parameters)
    grad_norm_after_clip = grad_norm_before_clip
    param_norm_before_clip = compute_param_norm(parameters)
 
    grad_before_rescale = rescale_gradients(model, grad_norm)
    #redundant code below. No need to use clip_grad_norm. It has already been clipped. 
    #Have not deleted it for reproducing the exact same numbers as reported in the paper.
    #Removing an instruction may change the random order in which data is read and hence may slightly, though statistically insignificantly, change the numbers.    
    if clip_val is not None:
        torch.nn.utils.clip_grad_norm(model.parameters(), clip_val)
    grad_norm_after_clip = compute_grad_norm(parameters)
    #
    optimizer.step()
    return grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip 
# == Helper functions for evaluation (some copied from Riedel EACL 2017)


def get_hierarchical_accuracy(gold_ids, predicted_ids, entity_hierarchy_linked_list):
    '''
        A relaxed evaluation for evaluating concept linking for hierarchically organised concepts
        NOTE: entity_hierarchy_linked_list is a linked list representation of the hierarchy and must not be the transitive closure of hierarchy
    '''
    batch_score = 0.0
    for gold_id, predicted_id in zip(gold_ids, predicted_ids):
        if gold_id == predicted_id:
            batch_score += 1.0
        elif gold_id in entity_hierarchy_linked_list:
            # if predicted ID is a direct parent of the gold ID
            if predicted_id in entity_hierarchy_linked_list[gold_id]:
                batch_score += 1.0
            # if predicted ID is a sibling of the gold ID
            elif predicted_id in entity_hierarchy_linked_list and len(entity_hierarchy_linked_list[predicted_id] & entity_hierarchy_linked_list[gold_id]) > 0:
                batch_score += 1.0

    return batch_score


def f1(p,r):
    if r == 0.:
        return 0.
    return 2 * p * r / float( p + r )


# ============= New class for scoring ===============
class Scorer(object):

    def __init__(self):
        self.acc_accum = 0.0
        self.num_seen = 0

        self.p_macro_accum = 0.0
        self.r_macro_accum = 0.0


        # micro numbers

        self.num_correct_labels = 0.0
        self.num_predicted_labels = 0
        self.num_true_labels = 0


    def get_scores(self):

        self.p_macro = self.p_macro_accum/self.num_seen
        self.r_macro = self.r_macro_accum/self.num_seen

        self.p_micro = self.num_correct_labels/self.num_predicted_labels
        self.r_micro = self.num_correct_labels/self.num_true_labels

        self.accuracy = self.acc_accum/self.num_seen
        self.macro_f1 = f1(self.p_macro, self.r_macro)
        self.micro_f1 = f1(self.p_micro, self.r_micro)
        return self.accuracy, self.macro_f1, self.micro_f1

    def get_scores_dict(self):
        return {'p_macro': self.p_macro, 'r_macro': self.r_macro, 'f_macro': self.macro_f1,'p_micro': self.p_micro, 'r_micro': self.r_micro, 'f_micro': self.micro_f1, 'acc': self.accuracy}

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def run(self, true_and_prediction):
        self.num_seen += len(true_and_prediction)

        for true_labels, predicted_labels in true_and_prediction:
            self.acc_accum += (set(true_labels) == set(predicted_labels))

            #update micro stats
            self.num_predicted_labels += len(predicted_labels)
            self.num_true_labels += len(true_labels)
            self.num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))

            #update macro stats
            if len(predicted_labels):
                self.p_macro_accum += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            if len(true_labels):
                self.r_macro_accum += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))





def strict(true_and_prediction):
    num_entities = len(true_and_prediction)
    correct_num = 0.
    for true_labels, predicted_labels in true_and_prediction:
        correct_num += set(true_labels) == set(predicted_labels)
    return correct_num

def loose_macro(true_and_prediction):
    num_entities = len(true_and_prediction)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1( precision, recall)

def loose_micro(true_and_prediction):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in true_and_prediction:
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1( precision, recall)


def AP(scores, gold_ids):
    '''
        Calculate the sum of the average precision for this batch
    '''
    aps = []
    for score,gold_id in zip(scores, gold_ids):
        aps.append(average_precision_score(gold_id, score))
    return sum(aps)



def margin_loss_linking(margin):
    '''
    compute the max margin score for entity linking
    scores are negative order violations
    '''

    def f(scores, pos_ids):
        scores_pos = scores.gather(1, pos_ids.unsqueeze(1)) #(batch_size, 1) all positive scores
        return (margin + scores - scores_pos).clamp(min=0).mean()

    return f


def margin_loss_typing(margin):
    '''
    compute the max margin score for typing when there are multiple positives
    '''

    def f(order_violations, pos_ids):
        num_batches = float(order_violations.data.shape[0])
        scores_pos = order_violations*pos_ids
        scores_neg = order_violations - scores_pos
        loss = scores_pos.sum() + (margin - scores_neg).clamp(min=0).sum()
        return loss/num_batches

    return f


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    #logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
