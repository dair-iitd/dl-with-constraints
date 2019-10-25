from collections import OrderedDict, defaultdict
import torch.nn.functional as F
import torch
import utils 
import settings 
from typing import Iterator, List, Dict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from penalty import BasePenalty

@Model.register("mtl_constraints")
class MTLConstraints(Model):
    def __init__(self, 
                vocab: Vocabulary,
                config: Dict):
        super().__init__(vocab)        
        self.constraint_dict = OrderedDict()
        #implication constraint
        self.constraint_dict['imp'] = MTLImplication(vocab, config['mtl_implication'])
       
        self.id2key = list(self.constraint_dict.keys())
        self.key2id = dict((k,i) for i,k in enumerate(self.id2key))
        self.dd_constant_lambda = config.get('dd_constant_lambda',False) 
        if self.dd_constant_lambda: 
            self.lambda_list = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(self.constraint_dict[k].num_constraints()),requires_grad=False) for k in self.id2key])
        else:
            self.lambda_list = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.constraint_dict[k].num_constraints())) for k in self.id2key])
        #self.lambda_dict = torch.nn.ParameterDict([(k,torch.nn.Parameter(torch.zeros(v.num_constraints()))) for k,v in self.constraint_dict.items()])
        if settings.cuda:
            self.lambda_list = self.lambda_list.cuda()
        #for k, v in self.constraint_dict.items():
        #    self.lambda_dict[k] = torch.nn.Parameter(
        #        torch.zeros(v.num_constraints()))


    def forward(self,task1_scores, task2_scores, mask):
        loss = 0.0
        penalties = {}
        for k in self.constraint_dict:
            penalty = (self.lambda_list[self.key2id[k]]*(self.constraint_dict[k].get_penalty(task1_scores,task2_scores,mask.float()).mean(dim=0))).sum()
            loss += self.constraint_dict[k].weight*penalty
            penalties[k+'_pen'] = penalty.item()

        penalties['loss'] = loss
        return penalties 

    def violation_stats(self,task1_scores,task2_scores,indicator=0):
        rs = OrderedDict()
        for k in self.constraint_dict:
            penalty = self.constraint_dict[k].get_penalty(task1_scores,task2_scores,indicator)
            if indicator == 0:
                k1 = k+'_v'
                k2 = k+'_tc'
                k3 = k+'_p'
            else:
                k1 = k + '_pv'
                k2 = k + '_ptc'
                k3 = k + '_pp'

            rs[k1] = (penalty > 0).sum().item()
            rs[k2] = penalty.numel()
            rs[k3] = penalty.sum().item() 
        return rs

    def lambda_stats(self):
        ls = OrderedDict()
        for k in self.constraint_dict:
            ls[k+'_nzl'] = (self.lambda_list[self.key2id[k]] > 0).float().sum().item()
            ls[k+'_tl'] = self.lambda_list[self.key2id[k]].numel()
            ls[k+'_apl'] = 0 if (ls[k+'_nzl'] == 0) else (self.lambda_list[self.key2id[k]].sum().item()/ls[k+'_nzl'])
            assert ((self.lambda_list[self.key2id[k]] < 0).float().sum().item() == 0)
        return ls


class MTLImplication(BasePenalty):
    def __init__(self,vocab, config):
        super(MTLImplication,self).__init__()
        self.weight = config['weight']
        self.populate_dd_vars(vocab,config['constraints_path'])

    def populate_dd_vars(self, vocab,constraints_path): #NOTE
        #taski_coefs.shape == #Tags x #Constraints
        self.task1_coefs, self.task2_coefs = utils.get_dd_coefs(constraints_path,vocab)
        self.nconstraints = self.task1_coefs.size(1)
            

    def num_constraints(self):
        return self.nconstraints 

    def get_penalty(self,task1_scores, task2_scores, mask,indicator=0):
        if indicator == 1:
            task1_prob = task1_scores
            task2_prob = task2_scores
        else:
            task1_prob = F.softmax(task1_scores,dim=2)
            task2_prob = F.softmax(task2_scores, dim=2)

        bsize = task1_prob.size(0)
        #cons_val: evaluate each constraint at each word.
        #cons_val.shape = bsize x w x self.num_constraints
        cons_val = torch.bmm(task1_prob, self.task1_coefs.unsqueeze(0).expand(bsize, -1, -1)) + \
        torch.bmm(task2_prob, self.task2_coefs.unsqueeze(0).expand(bsize, -1, -1))
        cons_val = F.relu(-1.0*cons_val)
        cons_val = cons_val*mask.unsqueeze(-1).expand_as(cons_val) 
        #cons_val = cons_val.sum(dim=1)/mask.sum(dim=1).unsqueeze(-1)
        cons_val = cons_val.sum(dim=1)
        #cons_val.shape  = bsize  x self.num_constraints
        return cons_val 


  
