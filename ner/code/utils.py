
import torch
import pickle
import numpy as np
import yaml
import os
import allennlp.nn.util as nn_util
import settings

def get_weights(vocab, namespace, eps=10):
    freq = torch.zeros(len(vocab.get_index_to_token_vocabulary(namespace)))
    if namespace in vocab._retained_counter:
        for key, value in vocab._retained_counter[namespace].items():
            freq[vocab.get_token_to_index_vocabulary(namespace)[key]] = value
    #
    freq = freq + eps
    wts = freq.sum()/freq
    return wts


def prepare_nerpos_data(input_file, output_file, shuffle=False, seed=914, stem=False):
    #input_file = '../data/ner-pos/gmb/train.txt'
    #output_file = '../data/ner-pos/gmb/train.pkl'
    fh = open(input_file)
    lines = fh.readlines()
    fh.close()
    sentences = []
    sentence = []
    for line in lines:
        if line.strip() == '':
            if len(sentence) > 0:
                sentences.append(sentence)
            sentence = []
        else:
            stem, pos, word, ner = line.strip().split()
            if stem:
                word = stem
            sentence.append([word, ner, pos])

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sentences)
    #

    dire = os.path.dirname(output_file)
    if not os.path.exists(dire):
        os.mkdir(dire)
    #
    ofh = open(output_file, 'wb')
    pickle.dump(sentences, file=ofh)
    ofh.close()



def prepare_conll_data(base_dir):
 for which_file,seed, shuffle in [('train.txt',2483,True),('dev.txt',0, False),('test.txt',0, False)]:
    prepare_nerpos_data(os.path.join(base_dir,which_file),os.path.join(base_dir,which_file+'.pkl'),shuffle,seed)
    for i,s in enumerate([2483, 7241, 6453, 4699, 5313, 779, 1825, 1599, 3121, 5126]):
        for which_file,seed, shuffle in [('train.txt',s,True),('dev.txt',0, False),('test.txt',0, False)]:
            prepare_nerpos_data(os.path.join(base_dir,which_file),os.path.join(base_dir,'shuffle'+str(i+1),which_file+'.pkl'),shuffle,seed)
        
   



"""

###Sample 1
for which_file,seed, shuffle in [('train.txt',2483,True),('dev.txt',0, False),('test.txt',0, False)]:
    utils.prepare_nerpos_data('../data/ner-pos/gmb/'+which_file,'../data/ner-pos/gmb/'+which_file+'.pkl',shuffle,seed)

import utils
for i,s in enumerate([2483, 7241, 6453, 4699, 5313, 779, 1825, 1599, 3121, 5126]):
    for which_file,seed, shuffle in [('train.txt',s,True),('dev.txt',0, False),('test.txt',0, False)]:
        utils.prepare_nerpos_data('../data/ner-pos/gmb/'+which_file,'../data/ner-pos/gmb/shuffle'+str(i+1)+'/'+which_file+'.pkl',shuffle,seed)
    

"""


def get_dd_coefs(constraints_path, vocab):
    con = yaml.load(open(constraints_path))
    num_constraints = len(con)
    num_ner_tags = vocab.get_vocab_size('task1_labels')
    num_pos_tags = vocab.get_vocab_size('task2_labels')

    ner_coefs = torch.zeros(num_ner_tags, num_constraints)
    pos_coefs = torch.zeros(num_pos_tags, num_constraints)
    for i, this_cons in enumerate(con):
        for ner_tag in con[i][0]:
            ner_coefs[vocab.get_token_index(ner_tag, 'task1_labels')][i] = -1
        #
        for pos_tag in con[i][1]:
            pos_coefs[vocab.get_token_index(pos_tag, 'task2_labels')][i] = 1


    if settings.cuda:
        return ner_coefs.cuda(), pos_coefs.cuda()
    else:
        return ner_coefs, pos_coefs


def dd_inference(ner_prob, pos_prob, mask, ner_coefs, pos_coefs, ddlr, 
        dditer, ner_metric=None, pos_metric=None, ner_labels=None, pos_labels=None):
    # ner_coefs.shape == ner_tags x num_constraints
    # pos_coefs = #pos_tags  x num_constraints
    bsize, num_words, _ = ner_prob.shape
    _, num_constraints = ner_coefs.shape
    lamdas = torch.zeros(bsize, num_words, num_constraints, device=ner_prob.device)
    num_violations = torch.zeros(dditer+1)
    if ner_labels is not None:
        ner_metric[0](ner_prob, ner_labels, mask)
    if pos_labels is not None:
        pos_metric[0](pos_prob, pos_labels, mask)

    ner_prob1 = ner_prob
    pos_prob1 = pos_prob
    for this_iter in range(dditer):
        # y = indicator variables for ner and z indicator variables for pos
        _, argmax_ner = torch.max(ner_prob1, dim=-1)
        y = torch.zeros(ner_prob1.shape,device=ner_prob1.device)
        
        y = y.scatter(2, argmax_ner.unsqueeze(-1), 1)

        _, argmax_pos = torch.max(pos_prob1, dim=-1)
        z = torch.zeros(pos_prob1.shape, device=pos_prob1.device)
        z = z.scatter(2, argmax_pos.unsqueeze(-1), 1)

        # evaluate constraints
        # y.shape == bsize,words,ner_tags,
        # ner_coefs.shape == ner_tags, num_constraints
        #cons_val.shape == bsize, words, num_constraints
        cons_val = torch.bmm(y, ner_coefs.unsqueeze(0).expand(bsize, -1, -1)) + \
            torch.bmm(z, pos_coefs.unsqueeze(0).expand(z.size(0), -1, -1))

        # its not a violation if cons_val > 0, hence need not update
        cons_val[cons_val > 0] = 0
        cons_val = cons_val*mask.float().unsqueeze(-1).expand_as(cons_val)
        
        this_violations = (cons_val < 0).sum().item()

        if this_violations == 0:
            break
        #

        num_violations[this_iter] = this_violations
        lamdas = lamdas - ddlr*cons_val 
        #    mask.float().unsqueeze(-1).expand_as(cons_val)

        #print(this_iter, this_violations,
        #     cons_val[255, 13, 1], lamdas[1, 13, 1])

        ner_prob1 = ner_prob + \
            torch.bmm(lamdas, ner_coefs.transpose(
                0, 1).unsqueeze(0).expand(bsize, -1, -1))
        pos_prob1 = pos_prob + \
            torch.bmm(lamdas, pos_coefs.transpose(
                0, 1).unsqueeze(0).expand(bsize, -1, -1))
        #

        #print(torch.argmax(ner_prob1[1,13]), ner_prob1[1,13])
        if ner_labels is not None:
            ner_metric[this_iter+1](ner_prob1, ner_labels, mask)
        if pos_labels is not None:
            pos_metric[(this_iter+1)](pos_prob1, pos_labels, mask)
    #

    for i in range(this_iter, dditer):
        num_violations[i+1] = this_violations
        if ner_labels is not None:
            ner_metric[i+1](ner_prob1, ner_labels, mask)
        if pos_labels is not None:
            pos_metric[(i+1)](pos_prob1, pos_labels, mask)
    #
    return ner_prob1, pos_prob1, this_iter+1, num_violations

