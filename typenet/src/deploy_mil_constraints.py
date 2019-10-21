'''
    Code for training/testing/evaluating/predicting joint linking + typing + structure.
    UNK always maps to 0
    Predictions are made only for freebase types, and hence, for ease we map all freebase types for ids first, and then map wordnet types
'''

#from IPython.core.debugger import Pdb
import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import *
from data_iterator import *
from build_data import *
from general_utils import *
import os
import torch.optim as optim
from config_mil import Config_MIL
from datetime import datetime as dt
import torch.nn.functional as F
import dd_utils
import constraints as constraints_module 
from collections import Counter, defaultdict, OrderedDict
import pandas as pd
import scheduler 
import math
import random

#logging.basicConfig(stream = sys.stdout, level=logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


slogger = logging.getLogger("stats")
slogger.setLevel(logging.INFO)
slogger.addHandler(console_handler)

logger = logging.getLogger("MIL Typing + Linking + structure")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

glogger = logging.getLogger("grad")
glogger.setLevel(logging.INFO)


def get_cooccuring_matrix(train_file,entity_type_dict,typenet_matrix):
    train_entities = read_entities(train_file)
    data = set()
    for e in train_entities:
        data.add(frozenset(entity_type_dict[e]))
    #
    coocur = dd_utils.get_cooccur(data)
    dd_pids, dd_cids = dd_utils.get_parent_child_tensors(typenet_matrix)
    dd_pids = dd_pids.numpy()
    dd_cids = dd_cids.numpy()
    ind = (dd_pids < coocur.shape[0]) & (dd_cids < coocur.shape[0])
    coocur[dd_pids[ind],dd_cids[ind]] += 1
    coocur[dd_cids[ind],dd_pids[ind]] += 1
    return coocur






def get_topk_mutl_excl_penalty(scores, dd_mut_excl1, dd_mut_excl2, config):
    penalty = dd_utils.get_mut_excl_penalty(scores, dd_mut_excl1, dd_mut_excl2,config)
    if (config.dd_mutl_excl_topk == -1) or (config.dd_mutl_excl_topk > dd_mut_excl1.shape[0]):
        return penalty
    else:
        penalty,_  = penalty.topk(config.dd_mutl_excl_topk,dim=1)
        return penalty



# == The main training/testing/prediction functions
def get_tensors_from_minibatch(minibatch, type_indexer, type_indexer_struct, typenet_matrix, num_iter, config, bag_size, train = True):
    mention_representation = np.array(minibatch.data['mention_representation']) # (batch_size*bag_size, embedding_dim)
    batch_size = len(mention_representation)/bag_size

    # possible mention types
    type_idx = type_indexer.repeat(batch_size*bag_size, 1) #(batch_size*bag_size, num_fb_types)
    gold_types = torch.FloatTensor(minibatch.data['gold_types']).view(-1, bag_size, config.fb_type_size)[:, 0, :] #(batch_size, num_types)
    mention_representation = torch.FloatTensor(mention_representation) #(batch_size*bag_size, embedding_dim)

    all_tensors = {'type_candidates' : type_idx, 'gold_types' : gold_types, 'mention_representation' : mention_representation}

    if config.struct_weight > 0 and train:
        # ========= STRUCTURE LOSS TENSORS ===========
        # get current batch of children types to optimize
        num_types = config.type_size
        st = (num_iter*batch_size)%num_types
        en = (st + batch_size)%num_types

        if en <= st:
            type_curr_child = range(st, num_types) + range(0, en)
        else:
            type_curr_child = range(st, en)

        type_curr_child = np.array(type_curr_child)
        batch_size_types = len(type_curr_child)


        all_tensors['structure_parents_gold'] = torch.FloatTensor(typenet_matrix[type_curr_child].reshape(batch_size_types, num_types)) # get parent of children sampled
        all_tensors['structure_parent_candidates'] = type_indexer_struct.repeat(batch_size_types, 1)
        all_tensors['structure_children'] = torch.LongTensor(type_curr_child.reshape(batch_size_types, 1)) # convert children sampled to tensors


        # == add entity to type structure loss if we have linking weights

        if config.linker_weight > 0:
            num_entities = config.entity_size
            ent_st = (num_iter*batch_size) % num_entities
            ent_en = (ent_st + batch_size) % num_entities

            if ent_en <= ent_st:
                sampled_child_nodes = range(ent_st, num_entities) + range(0, ent_en)
            else:
                sampled_child_nodes = range(ent_st, ent_en)

            batch_size_struct   = len(sampled_child_nodes)
            parent_candidates   = []

            parent_candidate_labels = []
            for node in sampled_child_nodes:
                curr_labels = []

                _node = inverse_entity_dict[node] #entity_type_dict maps entity lexical form to a set of types

                parent_candidates_curr = np.random.choice(num_types, config.parent_sample_size, replace = False) # choose a random set of parent links
                for parent_candidate in parent_candidates_curr:
                    if _node not in entity_type_dict:
                        curr_labels.append(0)
                    elif parent_candidate in entity_type_dict[_node]:
                        curr_labels.append(1)
                    else:
                        curr_labels.append(0)

                parent_candidate_labels.append(curr_labels)
                parent_candidates.append(parent_candidates_curr)

            all_tensors['structure_parents_gold_entity'] = torch.FloatTensor(parent_candidate_labels) # (batch_size_struct, config.parent_sample_size)
            all_tensors['structure_parent_candidates_entity'] = torch.LongTensor(parent_candidates)   # (batch_size_struct, config.parent_sample_size)
            all_tensors['structure_children_entity']  = torch.LongTensor(sampled_child_nodes).view(batch_size_struct, 1) # (batch_size_struct, 1)



    if config.encoder == "rnn_phrase":
        all_tensors['st_ids'] = torch.LongTensor(minibatch.data['st_ids'])
        print("s1", all_tensors['st_ids'].shape)
        all_tensors['en_ids'] = torch.LongTensor(minibatch.data['en_ids'])

    all_tensor_names = all_tensors.keys()
    for tensor_name in all_tensors.keys():
        if config.gpu:
            all_tensors[tensor_name] = Variable(all_tensors[tensor_name].cuda())
        else:
            all_tensors[tensor_name] = Variable(all_tensors[tensor_name])


    if config.linker_weight > 0:
        # ========= LINKING LOSS tensors ============
        all_tensors['entity_candidates'], all_tensors['entity_candidate_lens'] = \
            pad_sequences(minibatch.data['entity_candidates'], PAD, torch.LongTensor, config.gpu) #(batch_size*bag_size, 100)

        all_tensors['priors'], _ = pad_sequences(minibatch.data['priors'], 0.0, torch.FloatTensor, config.gpu) #(batch_size*bag_size, 100)


    # == feed to mention typer and get scores ==
    sentences , _ = pad_sequences(minibatch.data['context'], PAD, torch.LongTensor, config.gpu)
    position_embeddings, _ = pad_sequences(minibatch.data['position_embeddings'], PAD, torch.LongTensor, config.gpu, pos = True)


    all_tensors['sentences'] = sentences #.view(batch_size, config.bag_size, -1)
    all_tensors['position_embeddings'] = position_embeddings  #.view(batch_size, config.bag_size, -1)

    return all_tensors


def train(model, train_data, dev, test, config, typenet_matrix, constraints, fb_type_size, unlabelled_data = None):
    start_time = dt.now()
    global_col_order = None
    # define the criterion for computing loss : currently only supports cross entropy
    #loss_criterion = nn.BCEWithLogitsLoss()
    loss_criterion = F.binary_cross_entropy_with_logits  
    loss_criterion_linking = nn.CrossEntropyLoss()


    # ===== Different L2 regularization on the structure weights =====

    base_parameters = []
    struct_parameters = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        print(name, param.is_leaf)
        if name.startswith('bilinear_matrix'):
            struct_parameters.append(param)
        else:
            base_parameters.append(param)

    if config.struct_weight > 0:
        optimizer = optim.Adam([{'params': struct_parameters, 'weight_decay' : config.bilinear_l2}, {'params': base_parameters}], lr=config.lr, eps=config.epsilon,
                           weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
    else:
        optimizer = optim.Adam(base_parameters, lr=config.lr, eps=config.epsilon,
                           weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
   
    dd_param_list = constraints.get_optim_params(config)
    dd_optim = None
    if len(dd_param_list) > 0:
        if config.dd_optim == 'adam':
            dd_optim = optim.Adam(dd_param_list, lr = config.ddlr,weight_decay = config.dd_weight_decay, betas = (config.beta1,config.beta2),eps=config.epsilon)
        else:
            dd_optim = optim.SGD(dd_param_list, lr = config.ddlr, momentum = config.dd_mom)
        #
    if dd_optim is not None:
        for x in dd_optim.param_groups:
            logger.info('lr: {}, len(params): {} ,  #each params: {}'.format(x['lr'], len(x['params']),  ','.join([str(y.numel()) for y in x['params']])))

    # transfer to GPU
    if config.gpu: 
        model.cuda()
        constraints.cuda()

    start_epoch, best_accuracy,num_iter,num_iter_total,lambda_iter,last_lambda_update, dd_update_freq = load_checkpoint(config,model,constraints,optimizer , dd_optim,config.start_from_best)
    torch.cuda.empty_cache()
    config.dd_update_freq = dd_update_freq 
    logger.info("Starting Values: #Supervised Iters: {}, #Lambda Iters: {}, #Last Lambda Update: {}, #Next After: {}".format(num_iter, lambda_iter, last_lambda_update, config.dd_update_freq)) 
    #override lr and mom
    if config.override_lr:
        logger.info("OVerriding lr")
        for x in optimizer.param_groups:
            x['lr'] = config.lr 
        
        if dd_optim is not None:
            dd_param_list = constraints.get_optim_params(config)
            for i,x in enumerate(dd_optim.param_groups):
                x['lr'] = dd_param_list[i]['lr']
            #
    #
    ddlrs_start = ''
    if dd_optim is not None:
        ddlrs_start = ','.join([str(x['lr']) for x in dd_optim.param_groups])
        
    lr_start = ','.join([str(x['lr']) for x in optimizer.param_groups])
    logger.info("Starting value of DDLR: {}. Optim LR: {}".format(ddlrs_start, lr_start))


    #lr scheduler
    #my_lr_scheduler = scheduler.CustomReduceLROnPlateau(optimizer, {'mode': 'min', 'factor': 0.1, 'patience': 7, 'verbose': True, 'threshold': 0.01, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0.00001, 'eps': 0.0000001}, maxPatienceToStopTraining=17) 
    my_lr_scheduler = scheduler.CustomReduceLROnPlateau(optimizer, {'mode': 'min', 'factor': 0.1, 'patience': 7, 'verbose': True, 'threshold': 0.01, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0.01*config.lr, 'eps': 0.0000001}, maxPatienceToStopTraining=17) 


    logger.info("\n==== Initial accuracy: %5.4f\n" %best_accuracy)

    num_fb_types = config.fb_type_size
    num_types = config.type_size

    type_indexer = torch.LongTensor(np.arange(num_fb_types)).unsqueeze(0)  #(1, num_fb_types) for indexing into the freebase part of type indexing array
    type_indexer_struct = torch.LongTensor(np.arange(num_types)).unsqueeze(0) #(1, num_types) for indexing into type embeddings array

    #num_iter = 0
    #num_iter_total = 0
    last_logged = -1
    #lambda_iter = 0
    patience = 0
    stop_training = False
    for epoch in xrange(start_epoch,config.num_epochs):
        logger.info("\n=====Epoch Number: %d =====\n" %(epoch+1))
        train_data.shuffle()
        generator_for_lambda,_   = get_mixer(train_data,unlabelled_data, id1 = 0, which_mixer = 'cm')
        if ((unlabelled_data is not None) and (num_iter < max(config.dd_warmup_iter,config.semi_warmup_iter)) and (not config.dd_constant_lambda)):
            #case when semi supervision is on but we are still warming up and hence dont want unlabeled data
            generator_for_wts, total_batches = get_mixer(train_data, None,id1 = 0)
        else:
            #case when either semi is off OR we are ready for consuming unlabeled data
            generator_for_wts,total_batches = get_mixer(train_data, unlabelled_data, id1 = 0, which_mixer = config.semi_mixer)
        labelled_id = 0
        for minibatch,ds_id in generator_for_wts:
            #print(num_iter,num_iter_total)
            all_tensors = get_tensors_from_minibatch(minibatch, type_indexer, type_indexer_struct, typenet_matrix, num_iter, config, config.bag_size)

            if config.encoder == "rnn_phrase":
                aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
            else:
                aux_data = all_tensors['position_embeddings']

            # set back to train mode
            model.train()
            constraints.train()
            struct_data  = [all_tensors['structure_children'], all_tensors['structure_parent_candidates']] if config.struct_weight > 0 else None
            struct_data_entity  = [all_tensors['structure_children_entity'], all_tensors['structure_parent_candidates_entity']] if (config.struct_weight > 0 and config.linker_weight > 0) else None
            linking_data = [all_tensors['entity_candidates'], all_tensors['priors']] if config.linker_weight > 0 else None
            logit_dict   = model(all_tensors['type_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], aux_data, config.bag_size, struct_data, linking_data, struct_data_entity)
            typing_loss = torch.tensor(0.0)
            if config.gpu:
                typing_loss = typing_loss.cuda()
            if ds_id == labelled_id:
                loss_wts = get_loss_weights(all_tensors['gold_types'], config)
                typing_loss = loss_criterion(logit_dict['typing_logits'], all_tensors['gold_types'],loss_wts)
            
            structure_loss = loss_criterion(logit_dict['type_structure_logits'], all_tensors['structure_parents_gold']) if config.struct_weight > 0 else 0.0
            structure_loss_entities = loss_criterion(logit_dict['entity_structure_logits'], all_tensors['structure_parents_gold_entity']) if (config.struct_weight > 0 and config.linker_weight > 0) else 0.0
            linker_loss  = loss_criterion_linking(logit_dict['linking_logits'], all_tensors['entity_candidate_lens'] - 1) if config.linker_weight > 0 else 0.0
            closs = torch.tensor(0)
            penalties = {}
            #break

            if config.dd:
                for lambda_param in constraints.parameters():
                    lambda_param.requires_grad = False
                #
                penalties = constraints(logit_dict['typing_logits'])
                closs = penalties.pop('loss')
            
            penalties['typing_loss'] = typing_loss.item()
            penalties['structure_loss'] = structure_loss.item() if isinstance(structure_loss,torch.Tensor) else structure_loss 
            train_loss = config.typing_weight*typing_loss + config.struct_weight*(structure_loss + structure_loss_entities) + config.linker_weight*linker_loss + config.dd_constraint_wt*closs 
            grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip = step(model, train_loss, optimizer, config.clip_val, config.grad_norm if ((num_iter >= config.dd_warmup_iter) or config.grad_norm_before_warmup) else None)
            num_iter += int(ds_id == labelled_id)
            num_iter_total += 1
            if num_iter_total % 250 == 0:
                logger.info("\n=====Loss: %5.4f DDLoss: %5.4f. Supervised Step: %d Total Step: %d Lambda Steps: %d ======\n" %(train_loss.item(), closs.item(), num_iter, num_iter_total, lambda_iter))


            if ((num_iter % 1000 == 0) and (last_logged != num_iter)):
                last_logged = num_iter 
                t1_ = dt.now()
                dev_stats,_  = evaluate(model, dev, config,constraints)   #"results_dev_%s.txt" %config_obj.model_file)
                dev_eval_time = (dt.now() - t1_).total_seconds()
                logger.info("+++++ Dev stats: {}".format(str(dev_stats)))
                logger.info('input to scheduler : {}'.format(1.0-1.0*dev_stats['map']))
                my_lr_scheduler.step(1.0-1.0*dev_stats['map'], epoch=num_iter//1000)

                t1_ = dt.now()
                test_eval_time = -1.0
                test_stats = {} 
                if ((dev_stats['map']  > best_accuracy) or (global_col_order is None)):
                    if (dev_stats['map'] > best_accuracy):
                        best_accuracy = dev_stats['map']
                        if config.save_model:
                            cpoint = generate_checkpoint(config,model,constraints,optimizer,dd_optim,epoch,num_iter,best_accuracy,is_best=True,num_iter_total=num_iter_total,lambda_iter = lambda_iter, last_lambda_update = last_lambda_update, dd_update_freq = config.dd_update_freq)
                            save_checkpoint(cpoint,epoch,True,config)

                    #test_stats,_  = evaluate(model, test, config,constraints)
                    test_stats,_  = evaluate(model, test, config,constraints, os.path.join(config.checkpoint_file,'test_sample_results.csv'),sample=0.01)
                    test_eval_time = (dt.now() - t1_).total_seconds()
                    logger.info("+++++ Test stats: {}".format(str(test_stats)))
                    patience = 0
                else:
                    patience += 1
                
                misc_stats = OrderedDict()
                misc_stats.update({'exp': config.model_name,'t': (dt.now() - start_time).total_seconds(), 'detime': dev_eval_time, 'tetime': test_eval_time, 'niter': num_iter, 'lr': optimizer.param_groups[0]['lr'], 'ntiter': num_iter_total, 'nliter': lambda_iter})
                misc_stats.update(penalties)
                
                dev_stats = dict(dev_stats)
                dev_stats['prefix'] = 'dev_'
                test_stats = dict(test_stats)
                test_stats['prefix'] = 'test_'
                
                dev_stats = add_prefix(dev_stats)
                test_stats = add_prefix(test_stats)
                misc_stats = add_prefix(misc_stats)

                if global_col_order is None:
                    global_col_order = misc_stats.keys()
                    global_col_order.sort()
                    dsk = dev_stats.keys()
                    dsk.sort()
                    global_col_order.extend(dsk)
                    tsk = test_stats.keys()
                    tsk.sort()
                    global_col_order.extend(tsk)
                    assert (len(global_col_order) == len(set(global_col_order)))
                    slogger.info(','.join(global_col_order))
                
                misc_stats.update(dev_stats)
                misc_stats.update(test_stats)
                slogger.info(','.join([str(round(misc_stats.get(k,-1), 6)) if isinstance(misc_stats.get(k,-1), float) else str(misc_stats.get(k,-1)) for k in global_col_order]))
                if (my_lr_scheduler.shouldStopTraining()):
                    logger.info("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(my_lr_scheduler.unconstrainedBadEpochs, my_lr_scheduler.maxPatienceToStopTraining))
                    stop_training = True 
                    break
  
            #num_iter_total : num_iter + unsupervised_iters
            lambda_grad_before_clip, lambda_grad_after_clip, lambda_norm  = -1, -1, -1
            if (config.dd and (not config.dd_constant_lambda)):
                #we want warmup with only supervised iterations
                #if ((num_iter_total % config.dd_update_freq == 0) and num_iter >= config.dd_warmup_iter):
                if (num_iter >= config.dd_warmup_iter) and (num_iter - last_lambda_update >= config.dd_update_freq):
                    #Pdb().set_trace()
                    model.eval()
                    constraints.train()
                    for lambda_param in constraints.parameters():
                        lambda_param.requires_grad = True
                    for minibatch,_ in generator_for_lambda:
                        with torch.no_grad():
                            all_tensors = get_tensors_from_minibatch(minibatch, type_indexer, None, None, 0, config, config.bag_size, train = False)
                            if config.encoder == "rnn_phrase":
                                aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
                            else:
                                aux_data = all_tensors['position_embeddings']
                            linking_data = [all_tensors['entity_candidates'], all_tensors['priors']] if config.linker_weight > 0 else None
                            logit_dict = model(all_tensors['type_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], aux_data, config.bag_size, None, linking_data)
                
                        #
                        closs = -1.0*config.dd_constraint_wt*constraints(logit_dict['typing_logits'].detach())['loss']
                        lambda_grad_before_clip, lambda_grad_after_clip, lambda_norm = step(constraints, closs,dd_optim) 
                        lambda_iter += 1
                        last_lambda_update = num_iter
                        
                        #increase dd_update_freq
                        if (config.dd_increase_freq_after is not None) and (lambda_iter % config.dd_increase_freq_after == 0):
                            config.dd_update_freq += config.dd_increase_freq_by 
                        #
                        lr_decay_after = 1.0
                        if hasattr(config,'dd_decay_lr_after'):
                            lr_decay_after = config.dd_decay_lr_after
                            assert lr_decay_after >= 0
                        #

                        if config.dd_decay_lr == 1: 
                            lr_decay_after = 1.0/lr_decay_after 
                            for param_group in dd_optim.param_groups:
                                param_group['lr'] = param_group['lr']*math.sqrt(float(lr_decay_after*(lambda_iter-1)+1)/float(lr_decay_after*lambda_iter+1))
                        elif config.dd_decay_lr == 2:
                            lr_decay_after = 1.0/lr_decay_after 
                            for param_group in dd_optim.param_groups:
                                param_group['lr'] = param_group['lr']*(float(lr_decay_after*(lambda_iter-1)+1)/float(lr_decay_after*lambda_iter+1))
                        elif config.dd_decay_lr == 3:
                            #exponential decay
                            assert lr_decay_after <= 1
                            for param_group in dd_optim.param_groups:
                                param_group['lr'] = param_group['lr']*lr_decay_after                            
                        # 
                        break
                    #
                    model.train()
                    if lambda_iter % 5 == 0:
                        logger.info("Updated Lambda. #Supervised Iters: {}, #Lambda Iters: {}, #Last Lambda Update: {}, #Next After: {}".format(num_iter, lambda_iter, last_lambda_update, config.dd_update_freq)) 
                        ddlrs_start = ','.join([str(x['lr']) for x in dd_optim.param_groups])
                        lr_start = ','.join([str(x['lr']) for x in optimizer.param_groups])
                        logger.info("DDLRs: {}. Optim LRs: {}".format(ddlrs_start, lr_start))
                #
                if num_iter == max(config.semi_warmup_iter, config.dd_warmup_iter):
                    logger.info("Break after warm up iters and start a new epoch")
                    break
            glogger.info(','.join(map(lambda x: str(round(x,6)),[num_iter, num_iter_total, lambda_iter, train_loss.item(), closs.item(), grad_norm_before_clip,grad_norm_after_clip,param_norm_before_clip,lambda_grad_before_clip,lambda_grad_after_clip, lambda_norm])))
        logger.info("Epoch # {} over".format(epoch))
        
        if config.save_model:
            cpoint = generate_checkpoint(config,model,constraints,optimizer,dd_optim,epoch,num_iter,best_accuracy,is_best=False,num_iter_total=num_iter_total,lambda_iter = lambda_iter, last_lambda_update = last_lambda_update, dd_update_freq = config.dd_update_freq)
            save_checkpoint(cpoint,epoch,False,config)
    
        if stop_training:
            break

    _ = load_checkpoint(config,model,best=True)
    torch.cuda.empty_cache()

    test_stats,_  = evaluate(model, test, config,constraints,os.path.join(config.checkpoint_file,'test_results.csv'))
    logger.info("Best Test Stats: {}".format(str(test_stats)))
    logger.info("\n===== Training Finished! =====\n")




def predict_linking(scores, entity_candidates):

    # scores is a batch_size x num_candidates tensor of logits.

    # == find the argmax and use that to index into entity_candidates ===
    _, idx = scores.max(dim = -1)
    return entity_candidates.gather(1, idx.unsqueeze(1)).squeeze(1)


#sscores == probability
def predict_typing_topk(scores, gold_ids, topk = 5):
    true_and_prediction = []
    topk_scores, topk_ind = scores.topk(k=topk)
    for ptag,true_label in zip(topk_ind.cpu().numpy(), gold_ids):
        predicted_tag = list(ptag)
        true_tag = list(np.where(true_label)[0])
        true_and_prediction.append((true_tag, predicted_tag))
    #
    return true_and_prediction





def predict_typing(scores, gold_ids, threshold=0.5):
    true_and_prediction = []
    for score,true_label in zip(scores, gold_ids):
        predicted_tag = []
        true_tag = []
        for label_id,label_score in enumerate(list(true_label)):
            if label_score > 0:
                true_tag.append(label_id)
        lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
        predicted_tag.append(lid)
        eps = 0.0000
        for label_id,label_score in enumerate(list(score)):
            if label_score >= min(ls - eps,threshold): #or label_score == score[lid]:
                if label_id != lid:
                    predicted_tag.append(label_id)
        true_and_prediction.append((true_tag, predicted_tag))

    return true_and_prediction

def evaluate(model, eval_data, config, constraints, log_file = None,sample = 1.0):
    threshold = config.inference_thr 
    _len = 0
    num_types = model.config.fb_type_size # only index into freebase partition
    _ap = 0.0
    test_bag_size = eval_data.bag_size
    
    # they go into a pandas table
    max_scores = []
    num_predictions = []
    sum_scores = []
    gold_count = []
    entity_list  = []
    
    model.eval()
    constraints.eval()
    return_stats = Counter() 
    scorer_obj  = Scorer()
    total_entities = 0
    total_labels = 0
    with torch.no_grad():
        if log_file is not None:
            f = open(log_file, "w")

        type_indexer = torch.LongTensor(np.arange(num_types)).unsqueeze(0)
        return_stats.update(constraints.lambda_stats())
        for minibatch in eval_data.get_batch():
            all_tensors = get_tensors_from_minibatch(minibatch, type_indexer, None, None, 0, config, test_bag_size, train = False)

            if config.encoder == "rnn_phrase":
                aux_data = (all_tensors['st_ids'], all_tensors['en_ids'])
            else:
                aux_data = all_tensors['position_embeddings']

            linking_data = [all_tensors['entity_candidates'], all_tensors['priors']] if config.linker_weight > 0 else None
            logit_dict = model(all_tensors['type_candidates'], all_tensors['mention_representation'], all_tensors['sentences'], aux_data, test_bag_size, None, linking_data)
            gold_ids = np.array(minibatch.data['gold_types']).reshape(-1, test_bag_size, num_types)[:, 0, :]
            scores = logit_dict['typing_logits']
            if config.eval_topk > 0:
                _,topk_ind = scores.topk(k = config.eval_topk)
                torch_predicted_ids = torch.zeros_like(scores).scatter(-1,topk_ind,1)
            else:
                thr = torch.clamp(scores.max(dim=1)[0],max=0.0)
                torch_predicted_ids = (scores >= thr[:,None]).float()
                
            total_entities += scores.shape[0]
            total_labels += gold_ids.sum()
            return_stats.update(constraints.violation_stats(scores))
            return_stats.update(constraints.violation_stats(torch_predicted_ids,indicator=1))
            gold_violations = constraints.violation_stats(torch.tensor(gold_ids).cuda(),indicator=1)
            gold_violations_ordered = OrderedDict()
            for key in gold_violations:
                gold_violations_ordered['gold_'+key] = gold_violations[key]

            return_stats.update(gold_violations_ordered)
            scores = F.sigmoid(logit_dict['typing_logits'])
            if config.eval_topk > 0:
                predicted_ids = predict_typing_topk(scores,gold_ids, topk = config.eval_topk)
            else:
                predicted_ids = predict_typing(scores.data.cpu().numpy(), gold_ids,threshold = threshold)

            scores = scores.data.cpu().numpy()
            _ap += AP(scores, gold_ids)

            scorer_obj.run(predicted_ids)
            _len += len(gold_ids)
            
            scores_all = F.sigmoid(logit_dict['typing_logits_all'])
            if (log_file is not None) and (np.random.rand(1)[0] < sample):
                write_to_file(f, minibatch, predicted_ids, scores, scores_all.data.cpu().numpy())
            
            max_scores.extend(list(scores.max(axis=1)))
            num_predictions.extend(list(torch_predicted_ids.sum(dim=1).cpu().numpy()))
            sum_scores.extend(list(scores.sum(axis=1)))
            gold_count.extend(list(gold_ids.sum(axis=1)))
            entity_list.extend(list(np.array(minibatch.data['ent'])[range(0,len(minibatch.data['ent']),eval_data.bag_size)]))

        sum_scores_gold_counts = None 
        #PLEASE REMOVE THE COMMENT. COMMENTING COZ OF SOME STUPID PANDAS BUG WHILE RUNNING ON SHAKUNTALA
        sum_scores_gold_counts = pd.DataFrame.from_dict({'entities': entity_list, 'ss': sum_scores, 'gc': gold_count,'ms': max_scores, 'np': num_predictions})

        if log_file is not None:
            #PLEASE REMOVE THE COMMENT. COMMENTING COZ OF SOME STUPID PANDAS BUG WHILE RUNNING ON SHAKUNTALA
            sum_scores_gold_counts.to_csv(log_file+'_scores_sum_gold_count.csv')
            f.close()

        _ = scorer_obj.get_scores()
        return_stats.update(scorer_obj.get_scores_dict())
        return_stats['map'] = _ap/float(_len)
        return_stats['te'] = total_entities
        return_stats['total_labels'] = total_labels
        return return_stats, sum_scores_gold_counts


def write_to_file(f, minibatch, predictions, scores, scores_all=None):
    sentences  = minibatch.data['context']
    gold_types = minibatch.data['gold_types']
    bag_size = len(sentences)/len(scores)
    for i in xrange(len(scores)):
        curr_entity   = minibatch.data['ent'][i*bag_size]
        curr_golds, _ = predictions[i]
        curr_gold_ids = map(lambda idx: inverse_type_dict[idx], curr_golds)
        sorted_scores = sorted(enumerate(scores[i]), key = lambda val : -1*val[1])
        predictions_curr = [ (inverse_type_dict[idx], score) for (idx, score) in sorted_scores ][:10]

        f.write("gold entity: %s\n" %curr_entity)
        f.write("gold types: %s\n" %(" ".join(curr_gold_ids)))
        f.write("Predicted gold types: %s\n" %(predictions_curr))
        f.write("------------------\n")
        for j in xrange(bag_size):
            curr_mention = " ".join([inverse_vocab_dict[_idx] for _idx in sentences[i*bag_size + j]])
            f.write("%s\n" %curr_mention)
            if scores_all is not None:
                sorted_scores_curr = sorted(enumerate(scores_all[i,j]), key = lambda val : -1*val[1])
                predictions_curr_mention = [ (inverse_type_dict[idx], score) for (idx, score) in sorted_scores_curr][:10]
                f.write("mention predictions: %s\n" %(predictions_curr_mention))
            
        f.write("=================\n")

def get_new_typeids(entity_type_dict, orig2new,num_fb_types):
    fb_entity_type_dict = {}
    for ent in entity_type_dict:
        curr = []
        for type_id in entity_type_dict[ent]:
            new_type_id = orig2new[type_id]
            if new_type_id < num_fb_types:
                curr.append(new_type_id)

            #orig_type = orig_idx2type[type_id]
            #if not orig_type.startswith("Synset"):
            #    curr.append(type2idx[orig_type])

        assert(len(curr) != 0)
        fb_entity_type_dict[ent] = set(curr) # easy to search
    return fb_entity_type_dict 


def filter_fb_types(type_dict, entity_type_dict,typenet_matrix_orig,typenet_adj_matrix_orig):
    fb_types = [_type for _type in type_dict if not _type.startswith("Synset")]
    wordnet_types = [_type for _type in type_dict if _type.startswith("Synset")]

    # reorder types to make fb types appear first
    all_types = fb_types + wordnet_types

    orig_idx2type = {idx : _type for (_type, idx) in type_dict.iteritems()}
    type2idx = {_type : idx for (idx, _type) in enumerate(all_types)}
    orig2new = {idx : type2idx[_type] for (_type,idx) in type_dict.iteritems()}
    typenet_matrix = np.zeros(typenet_matrix_orig.shape)
    for i,j in zip(*np.where(typenet_matrix_orig == 1)):
        typenet_matrix[orig2new[i],orig2new[j]] = 1

    typenet_adj_matrix = np.zeros(typenet_adj_matrix_orig.shape)
    for i,j in zip(*np.where(typenet_adj_matrix_orig == 1)):
        typenet_adj_matrix[orig2new[i],orig2new[j]] = 1


    # 1. filter out only fb types and 2. change fb IDs according to type2idx
    fb_entity_type_dict = {}
    for ent in entity_type_dict:
        curr = []
        for type_id in entity_type_dict[ent]:
            orig_type = orig_idx2type[type_id]
            if not orig_type.startswith("Synset"):
                curr.append(type2idx[orig_type])

        assert(len(curr) != 0)
        fb_entity_type_dict[ent] = set(curr) # easy to search


    return type2idx, fb_entity_type_dict, len(fb_types), typenet_matrix, typenet_adj_matrix, orig2new


'''
Parse inputs
'''
def get_params():
    parser = argparse.ArgumentParser(description = 'Entity linker')
    parser.add_argument('-dataset', action="store", default="ACE", dest="dataset", type=str)
    parser.add_argument('-features', action = "store", default = 0, dest = "features", type=int)
    parser.add_argument('-model_name', action="store", default="entity_linker", dest="model_name", type=str)
    parser.add_argument('-dropout', action="store", default=0.5, dest="dropout", type=float)
    parser.add_argument('-train', action="store", default=1, dest="train", type=int)
    parser.add_argument('-bag_size', action="store", default=10, dest="bag_size", type=int)
    parser.add_argument('-encoder', action="store", default="position_cnn", dest="encoder", type=str)
    parser.add_argument('-asymmetric', action = "store", default=0, dest="asymmetric", type=int)
    parser.add_argument('-struct_weight', action = "store", default = 0, dest="struct_weight", type=float)
    parser.add_argument('-linker_weight', action = "store", default = 0, dest = "linker_weight", type=float)
    parser.add_argument('-typing_weight', action = "store", default = 0, dest = "typing_weight", type=float)
    parser.add_argument('-mode', action = "store", default = "typing", dest = "mode", type=str)
    parser.add_argument('-bilinear_l2', action = "store", default = 0.0, dest = "bilinear_l2", type=float)
    parser.add_argument('-parent_sample_size', action = "store", default = 100, dest="parent_sample_size", type=int)
    parser.add_argument('-complex', action = "store", default = 0, dest = "complex", type=int)


    parser.add_argument('-base_dir', action="store", default="/iesl/canvas/smurty/epiKB", type=str)
    parser.add_argument('-lr', action="store", default=1e-3, type=float)
    parser.add_argument('-beta1', action="store", default=0.9, type=float)
    parser.add_argument('-beta2', action="store", default=0.999, type=float)
    parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
    parser.add_argument('-weight_decay', action="store", default=0.0, type=float)
    parser.add_argument('-save_model', action="store", default=1, type=int)
    parser.add_argument('-clip_val', action="store", default=5, type=int)
    parser.add_argument('-embedding_dim', action="store", default=300, type=int)
    parser.add_argument('-hidden_dim', action="store", default=150, type=int)
    parser.add_argument('-num_epochs', action="store", default=2, type=int)
    parser.add_argument('-kernel_width', action="store", default=5, type=int)
    parser.add_argument('-batch_size', action="store", default=256, type=int)
    parser.add_argument('-test_batch_size', action="store", default=1024, type=int)
    parser.add_argument('-take_frac', action="store", default=1.0, type=float)
    parser.add_argument('-use_transitive', action="store", default=1, type=int)
    parser.add_argument('-log_dir', action="store", default="../typenetlogs", type=str)
    parser.add_argument('-ddlr',type=float,default = 0.001)
    parser.add_argument('-dd_constraint_wt',type=float,default = 1.0)
    parser.add_argument('-dd',action='store_true')
    parser.add_argument('-dd_update_freq',type=int, default = 10)
    parser.add_argument('-dd_warmup_iter',type=int, default = 1000)
    parser.add_argument('-dd_tcc',type=int, default = 0)
    parser.add_argument('-dd_weight_decay',type=float, default = 0.0)
    parser.add_argument('-dd_optim',type=str, default = 'adam')
    parser.add_argument('-dd_mom',type=float, default = 0.0)
    parser.add_argument('-dd_logsig',type=int, default = 1)
    parser.add_argument('-dd_penalty',type=str, default = 'strict')

    parser.add_argument('-dd_implication_wt',type=float,default = 1.0)
    parser.add_argument('-dd_mutl_excl_wt',type=float,default = 1.0)
    parser.add_argument('-dd_maxp_wt',type=float,default = 1.0)
    parser.add_argument('-dd_maxp_thr',type=float,default = 1.0)
    parser.add_argument('-use_wt_as_lr_factor',type=int,default = 0)

    parser.add_argument('-dd_increase_freq_by',type=int,default = 0)
    parser.add_argument('-dd_increase_freq_after',type=int,default = None)
    parser.add_argument('-dd_decay_lr',type=int,default = 3)
    parser.add_argument('-dd_decay_lr_after',type=float,default = 1.0, help='decay dd lr after how many lambda updates? Or memory in case of exp decay')

    parser.add_argument('-dd_mutl_excl_topk',type=int,default = -1)
    parser.add_argument('-inference_thr',type=float, default = 0.5)
    parser.add_argument('-class_imb',type=float, default = 0)
    parser.add_argument('-dd_constant_lambda',type=int, default = 0)
    parser.add_argument('-semi_mixer',type=str, default = 'bm')
    parser.add_argument('-semi_sup',type=int, default = 0)
    parser.add_argument('-unlabelled_ratio',type=float, default = 2.0)
    parser.add_argument('-semi_warmup_iter',type=float, default = 1000)
    parser.add_argument('-override_lr',type=int, default = 0)
    parser.add_argument('-start_from_best',type=int, default = 1)
    
    parser.add_argument('-random_seed',type=int, default = 1985)
    parser.add_argument('-numpy_seed',type=int, default = 14)
    parser.add_argument('-pytorch_seed',type=int, default = 28)
   
    parser.add_argument('-grad_norm',type=float, default = None)
    parser.add_argument('-grad_norm_before_warmup',type=int, default = 1)
    opts = parser.parse_args(sys.argv[1:])
    for arg in vars(opts):
        print arg, getattr(opts, arg)
    return opts


def read_entities(filename, start = 0.0, end = 1.0):
    f = open(filename)
    entities = set()

    for line in f:
        line = line.strip().split("\t")
        entities.add(line[0])

    start_idx = int(start*len(entities))
    end_idx  = int(end*len(entities))
    return set(list(entities)[start_idx: end_idx])


def set_random_seeds(opts):
    seed = opts.random_seed 
    numpy_seed = opts.numpy_seed
    torch_seed = opts.pytorch_seed
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


if __name__ == "__main__":

    opts = get_params()
    set_random_seeds(opts)
    run_dir = os.getcwd()
    config_obj = Config_MIL(run_dir, opts)

    formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt = '%Y%m%d %H:%M:%S')

    fhandle = logging.FileHandler(os.path.join(config_obj.checkpoint_file,'logs.txt'))
    fhandle.setFormatter(formatter)
    logger.addHandler(fhandle)
    logger.info(opts)
    
    stats_handle = logging.FileHandler(os.path.join(config_obj.checkpoint_file,'stats.csv'))
    slogger.addHandler(stats_handle)
    grad_handle = logging.FileHandler(os.path.join(config_obj.checkpoint_file, 'grad.csv'))
    glogger.addHandler(grad_handle)
    glogger.propagate = False
    glogger.info('num_iter, num_iter_total, lambda_iter, train_loss, closs, grad_norm_before_clip,grad_norm_after_clip,param_norm_before_clip,lambda_grad_before_clip,lambda_grad_after_clip, lambda_norm')
    #slogger.info('t,detime,tetime,exp,niter,dap,tap,tloss,ddloss,pen,lamda,nzl,nzml,tlam,tmelam,dsv,drv,dpv,dmev,de,dc,dmet,tsv,trv,tpv,tmev,te,tc,tmet,tsa,dsa,dmaf,dmif,tmaf,tmif')


    assert(config_obj.mode in ['typing', 'linking'])

    # === load in all the auxiliary data
    type_dict = joblib.load(config_obj.type_dict)
    entity_type_dict = joblib.load(config_obj.entity_type_dict)
    entity_dict  = joblib.load(config_obj.entity_dict)

    typenet_matrix_orig = joblib.load(config_obj.typenet_matrix)
    typenet_adj_matrix_orig = joblib.load(config_obj.typenet_adj_matrix)

    type_dict, entity_type_dict, fb_type_size,typenet_matrix, typenet_adj_matrix,map_old_to_new = filter_fb_types(type_dict, entity_type_dict, typenet_matrix_orig, typenet_adj_matrix_orig)
    if config_obj.use_transitive:
        entity_type_dict_test = entity_type_dict
    else:
        logger.info("Not Using Transitive Closure Labels")
        entity_type_dict_test_orig = joblib.load(config_obj.entity_type_dict_test)
        entity_type_dict_test = get_new_typeids(entity_type_dict_test_orig,map_old_to_new,fb_type_size)

    #typenet_matrix = typenet_matrix_orig 
    #typenet_adj_matrix = typenet_adj_matrix_orig 
   
    
    vocab_dict = joblib.load(config_obj.vocab_file)
    config_obj.vocab_size   = len(vocab_dict)
    config_obj.type_size    = len(type_dict)
    config_obj.entity_size  = len(entity_dict)

    config_obj.fb_type_size = fb_type_size

    inverse_vocab_dict = {idx: word for (word, idx) in vocab_dict.iteritems()}
    inverse_type_dict = {idx: word for (word, idx) in type_dict.iteritems()}
    inverse_entity_dict = {idx : word for (word, idx) in entity_dict.iteritems()}

    logger.info("\nNumber of words in vocab: %d\n" %config_obj.vocab_size)
    logger.info("\nNumber of total types in vocab: %d\n" %config_obj.type_size)
    logger.info("\nNumber of total entities in vocab: %d\n" %config_obj.entity_size)
    logger.info("\nNumber of fb types in vocab: %d\n" %config_obj.fb_type_size)


    train_entities = read_entities(config_obj.train_file, end = config_obj.take_frac)
    cooccur = get_cooccuring_matrix(config_obj.train_file,entity_type_dict,typenet_matrix)
    typenet_constraints = constraints_module.get_constraints(config_obj,{'cooccur': cooccur, 'adj_matrix': typenet_matrix[:fb_type_size,:fb_type_size]})

    pretrained_embeddings = get_trimmed_glove_vectors(config_obj.embedding_file)

    #=== now load in crosswikis
    if config_obj.linker_weight > 0:
        time_st = time.time() 
        alias_table = joblib.load(config_obj.cross_wikis_shelve)
        time_en = time.time()
        print("Time taken for reading alias table: %5.4f" %(time_en - time_st))

    else:
        alias_table = {}

    
    attribs = ['mention_representation', 'context', 'position_embeddings', 'gold_types', 'ent', 'st_ids', 'en_ids', 'entity_candidates', 'priors', 'gold_ids']

    if config_obj.encoder == "position_cnn":
        print("Using position embeddings with CNN")
        encoder = MentionEncoderCNNWithPosition(config_obj, pretrained_embeddings)
    elif config_obj.encoder == "basic":
        print("using regular CNNs")
        encoder = MentionEncoderCNN(config_obj, pretrained_embeddings)
    elif config_obj.encoder == "rnn_phrase":
        print("using a deep GRU and phrase embeddings")
        encoder = MentionEncoderRNN(config_obj, pretrained_embeddings)

    if config_obj.complex:
        model = MultiInstanceTyper_complex(config_obj, encoder)
    else:
        model   = MultiInstanceTyper(config_obj, encoder)

        

    # === Define the training and dev data

    train_bags = joblib.load(config_obj.bag_file)
    dev_bags   = joblib.load(config_obj.bag_file)
    test_bags  = joblib.load(config_obj.bag_file)

    unlabelled_entities = read_entities(config_obj.train_file, start = config_obj.take_frac,end= config_obj.unlabelled_ratio*config_obj.take_frac+config_obj.take_frac)
    dev_entities   = read_entities(config_obj.dev_file)
    test_entities  = read_entities(config_obj.test_file)


    train_data = MILDataset(train_bags, entity_type_dict, attribs, train_entities,vocab_dict, pretrained_embeddings, config_obj.encoder, config_obj.batch_size, config_obj.fb_type_size, MILtransformer, config_obj.bag_size, entity_dict, alias_table, typenet_matrix, True)
    unlabelled_data = None
    if config_obj.semi_sup:
        unlabelled_data = MILDataset(train_bags,None, attribs, unlabelled_entities, vocab_dict, pretrained_embeddings, config_obj.encoder,  config_obj.batch_size, config_obj.fb_type_size, MILtransformer, config_obj.bag_size, entity_dict,  alias_table, typenet_matrix,True)


    dev_data   = MILDataset(dev_bags, entity_type_dict_test, attribs, dev_entities,  vocab_dict, pretrained_embeddings, config_obj.encoder, config_obj.test_batch_size, config_obj.fb_type_size, MILtransformer, config_obj.test_bag_size, entity_dict, alias_table, typenet_matrix, False)
    test_data  = MILDataset(test_bags,entity_type_dict_test, attribs, test_entities, vocab_dict, pretrained_embeddings, config_obj.encoder, config_obj.test_batch_size, config_obj.fb_type_size, MILtransformer, config_obj.test_bag_size, entity_dict, alias_table, typenet_matrix, False)


    
    if opts.train:
        logger.info("\n====================TRAINING STARTED====================\n")
        train(model, train_data, dev_data, test_data, config_obj, typenet_matrix,constraints = typenet_constraints,fb_type_size = fb_type_size, unlabelled_data = unlabelled_data)
    else:
        logger.info("\n====================EVALUATION STARTED====================\n")
          
        if config_obj.gpu:
            model.cuda()
            typenet_constraints.cuda()
        #start_epoch, best_score,num_iter,num_iter_total,lambda_iter = load_checkpoint(config_obj,model,typenet_constraints,best=True) 
        start_epoch, best_score,num_iter,num_iter_total,lambda_iter,last_lambda_update, dd_update_freq = load_checkpoint(config_obj,model,typenet_constraints,best=True)
        torch.cuda.empty_cache()
        test_stats,test_score_stats = evaluate(model, test_data, config_obj,typenet_constraints, os.path.join(config_obj.checkpoint_file,'test_results_saved.txt')) #"results_test_%s.txt" %config_obj.model_file)

        dev_stats,dev_score_stats = evaluate(model, dev_data, config_obj,typenet_constraints)
        train_stats, train_score_stats = evaluate(model, train_data, config_obj, typenet_constraints)
        logger.info("Test STats: {}".str(test_stats))





if False:
    print(adf)
