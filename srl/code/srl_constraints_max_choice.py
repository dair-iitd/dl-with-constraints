from collections import OrderedDict, defaultdict
import torch.nn.functional as F
import torch
import settings 
from typing import Iterator, List, Dict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from penalty import BasePenalty, SpanImplicationMax, TerminalTag, Transition 
from allennlp.modules.conditional_random_field import allowed_transitions
from IPython.core.debugger import Pdb

@Model.register("srl_constraints_cagg_choice")
class SRLConstraintsCustomAggChoice(Model):
    def __init__(self, 
                vocab: Vocabulary,
                config: Dict):
        super().__init__(vocab)        
        self.constraint_dict = OrderedDict()
        #Pdb().set_trace()
        #implication constraint
        bl_config = config.get('span_implication_bl',None)
        lb_config = config.get('span_implication_lb',None)
        b_config = config.get('terminal_tag_b',None)
        l_config = config.get('terminal_tag_l',None)

        if bl_config is not None:
            self.constraint_dict['BL'] = SpanImplicationMax(vocab, bl_config, "BL")

        if lb_config is not None:
            self.constraint_dict['LB'] = SpanImplicationMax(vocab, lb_config, "LB")
    
        if b_config is not None:
            self.constraint_dict['B'] = TerminalTag(vocab, b_config, "B")

        if l_config is not None:
            self.constraint_dict['L'] = TerminalTag(vocab, l_config, "L") 

        config_transition_constraints = config.get('transition', None)
        if config_transition_constraints is not None:
            self.constraint_dict['transition'] = Transition(vocab, config_transition_constraints)
       
        self.id2key = list(self.constraint_dict.keys())
        self.key2id = dict((k,i) for i,k in enumerate(self.id2key))
        self.dd_constant_lambda = config.get('dd_constant_lambda',False) 
        if self.dd_constant_lambda: 
            #self.lambda_list = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(self.constraint_dict[k].num_constraints())) for k in self.id2key],requires_grad = False)
            self.lambda_list = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(self.constraint_dict[k].num_constraints()),requires_grad=False) for k in self.id2key])
        else:
            self.lambda_list = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.constraint_dict[k].num_constraints())) for k in self.id2key])
        #self.lambda_dict = torch.nn.ParameterDict([(k,torch.nn.Parameter(torch.zeros(v.num_constraints()))) for k,v in self.constraint_dict.items()])
        if settings.cuda:
            self.lambda_list = self.lambda_list.cuda()
        #for k, v in self.constraint_dict.items():
        #    self.lambda_dict[k] = torch.nn.Parameter(
        #        torch.zeros(v.num_constraints()))

    def forward(self, scores, mask, span_matrix,debug=False):
        loss = 0.0
        penalties = {}
        for k in self.constraint_dict:
            #Pdb().set_trace()
            violations,debug_out = self.constraint_dict[k].get_penalty(scores,mask.float(),span_matrix.float(),debug = debug)
            penalty = (self.lambda_list[self.key2id[k]]*violations.mean(dim=0)).sum()
            if debug:
                penalties[k+'_violations'] = debug_out.sum(dim=-1).detach()
            #
            loss += self.constraint_dict[k].weight*penalty
            penalties[k+'_pen'] = penalty.item()

        penalties['loss'] = loss
        return penalties 

    def violation_stats(self,scores,indicator=0):
        rs = OrderedDict()
        for k in self.constraint_dict:
            penalty = self.constraint_dict[k].get_penalty(scores,indicator)
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

# The mean - NOT per word BUT per sentence
"""
# Span constraints
class SpanImplicationMax(BasePenalty):
    def __init__(self, vocab, config, spanType: str):
        super(SpanImplicationMax,self).__init__()
        # TODO - is the weight supposed to be default value 1?
        self.weight = 1
        self.spanType = spanType        
        try:
            self.weight = config['weight']
        except:
            pass
        if (spanType not in ["BL", "LB"]):
            raise Exception("SpanImplication constraint: spanType should be BL or LB")
        self.populate_dd_vars(vocab, self.spanType, config)

        if settings.cuda:
            self.Btags_tensor = self.Btags_tensor.cuda()
            self.Ltags_tensor = self.Ltags_tensor.cuda()

        # Also initialize tensor of B and L tags        

    # In config, add an option - "aggregate_type" - can be max, sum etc
    def populate_dd_vars(self, vocab, spanType, config): #NOTE

        Btag = "B"
        Ltag = "L"
        if(spanType == "LB"):
            Btag = "L"
            Ltag = "B"
        self.Btags_tensor = None
        self.Ltags_tensor = None
        Btags_dict = {}
        Ltags_dict = {}    
        
        index2tagtype = [] # List the tag types        

        index2labels = vocab.get_index_to_token_vocabulary(namespace="labels")

        self.nconstraints = 0

        for idx in index2labels:
            lbl = index2labels[idx]
            if(lbl != 'O' and lbl[2:] != 'V'):
                l = lbl[2:]
                posn = lbl[0]
                if (l not in index2tagtype):
                    index2tagtype.append(l)

                if posn == Btag:
                    Btags_dict[l] = idx
                elif posn == Ltag:
                    Ltags_dict[l] = idx

        # Now form Btags_tensor and Ltags_tensor
        self.Btags_tensor, self.Ltags_tensor = [], []
        for idx, tagtype in enumerate(index2tagtype):
            self.Btags_tensor.append(Btags_dict[tagtype])
            self.Ltags_tensor.append(Ltags_dict[tagtype])

        self.Btags_tensor = torch.LongTensor(self.Btags_tensor)
        self.Ltags_tensor = torch.LongTensor(self.Ltags_tensor)

        self.nconstraints = len(index2tagtype)
            
        # aggregate_type
        aggregate_type = config.get("aggregate_type", "max")
        # max penalty over all expressions in  the logic formula (Yatin's formula)
        self.use_maxp = config.get("use_maxp", False)

        if self.use_maxp == True:
	        self.max_penalty = config.get("max_penalty", "log")
	        if self.max_penalty not in ["log", "negate"]:
	        	raise Exception("Penalty type for max should be log or negate")

        if aggregate_type.lower() not in ["sum", "max"]:
        	raise Exception("SpanImplicationMax aggregation type invalid")

        if aggregate_type.lower() == "sum":
        	self.aggregate_mat = lambda x: x.sum(dim=2)
        elif aggregate_type.lower() == "max":
        	self.aggregate_mat = lambda x: x.max(dim=2)[0]

    def num_constraints(self):
        return self.nconstraints 

    # Pass in class PROBABILITIES, NOT logits!
    def get_penalty(self, prob, mask, span_matrix, indicator=0):
        # prob is class PROBABILITIES, NOT logits(keep the option?)

        # prob - bsize x senlen x ntags
        # span_matrix - bsize x senlen x senlen        

        # NOTE - the summation can be replaced by max later if we want
        # Here we take NEGATIVE of expression so we can do hinge on it
        # constr_mat - bsize x senlen x |Btags_tensor|
        #mask out the span matrix
    
        if settings.cuda:
            span_matrix = span_matrix.cuda()
        #
        span_matrix = span_matrix*mask.unsqueeze(-1).expand_as(span_matrix)
        span_matrix = span_matrix*mask.unsqueeze(1).expand_as(span_matrix)

        if(self.spanType == "LB"):
            local_span_matrix = span_matrix.transpose(1,2)
        else:
            local_span_matrix = span_matrix

        if settings.cuda:
            local_span_matrix = local_span_matrix.cuda()

        # local_span_matrix - bsize x senlen x senlen
        # prob[:, :, self.Ltags_tensor] - bsize x senlen x ntags
        bsize,senlen,_ = prob.size() 
        ntags = self.nconstraints 
        #Pdb().set_trace() 
        expanded_span_mat = local_span_matrix.unsqueeze(-1).expand(bsize, senlen, senlen, ntags)
        expanded_ltags_mat = prob[:, :, self.Ltags_tensor].unsqueeze(1).expand(bsize, senlen, senlen, ntags)

        #constr_mat.shape  = B x S x S x nconstraints
        constr_mat = expanded_span_mat * expanded_ltags_mat

        active_span_mat = local_span_matrix.sum(dim=2)
        active_span_mat[active_span_mat > 0] = 1

        if self.use_maxp == False:
            constr_mat = prob[:, :, self.Btags_tensor] - self.aggregate_mat(constr_mat)
        else:
            constr_mat = torch.max(1-prob[:, :, self.Btags_tensor], (constr_mat.max(dim=2)[0]))
            constr_mat[active_span_mat.unsqueeze(-1).expand_as(constr_mat) == 0] = 1
            if self.max_penalty == "log":
            	constr_mat = -1.0 * torch.log(constr_mat)
            elif self.max_penalty == "negate":
            	constr_mat = 1.0 - constr_mat

        # constr_mat = prob[:, :, self.Btags_tensor] - torch.bmm(local_span_matrix, prob[:, :, self.Ltags_tensor])
        # active_span_mat - bsize x senlen - Check which spans are inactive        
        constr_mat = constr_mat * active_span_mat.unsqueeze(-1).expand_as(constr_mat) # Shapes match?
        #mask it again
        constr_mat =  constr_mat*mask.unsqueeze(-1).expand_as(constr_mat)
        F.relu(constr_mat, inplace=True)
        constr_mat = constr_mat.sum(dim=1)
        return constr_mat


# B/L constraint - can't be one if no span is present
class TerminalTag(BasePenalty):
    def __init__(self, vocab, config, terminalType: str):
        super(TerminalTag,self).__init__()
        # TODO - is the weight supposed to be default value 1?
        self.weight = 1
        self.terminalType = terminalType
        try:
            self.weight = config['weight']
        except:
            pass

        if terminalType not in ["B", "L"]:
            raise Exception("TerminalTag constraint: Tag must be B- or L- type.")

        self.populate_dd_vars(vocab, terminalType)
        # Also initialize tensor of B and L tags  
        if settings.cuda:
            self.Ttags_tensor = self.Ttags_tensor.cuda()            

    def populate_dd_vars(self, vocab, terminalType): #NOTE

        self.Ttags_tensor = None

        Ttags_dict = {}
        index2tagtype = [] # List the tag types

        index2labels = vocab.get_index_to_token_vocabulary(namespace="labels")

        self.nconstraints = 0

        for idx in index2labels:
            lbl = index2labels[idx]
            if(lbl != 'O' and lbl[2:] != 'V'):
                l = lbl[2:]
                posn = lbl[0]
                if (l not in index2tagtype):
                    index2tagtype.append(l)

                if posn == terminalType:
                    Ttags_dict[l] = idx

        # Now form Ttags_tensor and Ltags_tensor
        self.Ttags_tensor = []
        for idx, tagtype in enumerate(index2tagtype):
            self.Ttags_tensor.append(Ttags_dict[tagtype])

        self.Ttags_tensor = torch.LongTensor(self.Ttags_tensor)

        self.nconstraints = len(index2tagtype)
            

    def num_constraints(self):
        return self.nconstraints 

    # Pass in class PROBABILITIES, NOT logits!
    def get_penalty(self, prob, mask, span_matrix, indicator=0):
        # prob is class PROBABILITIES, NOT logits(keep the option?)

        # prob - bsize x senlen x ntags
        # span_matrix - bsize x senlen x senlen
        # constr_mat - bsize x sellen

        # active_span_mat - bsize x senlen - Check which spans are inactive
        if (self.terminalType == "B"):
            local_span_matrix = span_matrix
        else:
            local_span_matrix = span_matrix.transpose(1,2)
        
        if settings.cuda:
            local_span_matrix = local_span_matrix.cuda()

        active_span_mat = local_span_matrix.sum(dim=2)        
        active_span_mat[active_span_mat > 0] = 1        
        constr_mat = prob[:, :, self.Ttags_tensor]
        # IMPORTANT - inactive spans are active for terminal constraint
        constr_mat = constr_mat * (1 - active_span_mat.unsqueeze(-1))
        constr_mat = constr_mat*mask.unsqueeze(-1).expand_as(constr_mat)
        constr_mat = constr_mat.sum(dim=1)  

        return constr_mat        

# Transition constraints
class Transition(BasePenalty):
    def __init__(self, vocab, config):
        super(Transition,self).__init__()
        # TODO - is the weight supposed to be default value 1?
        self.weight = config.get("weight", 1)        

        self.config = config

        self.populate_dd_vars(vocab)
        
        # Put tensors on CUDA
        if settings.cuda:
            self.transition_matrix = self.transition_matrix.cuda()            

    def populate_dd_vars(self, vocab): #NOTE

        # Set up allowed transitions        
        all_labels = vocab.get_index_to_token_vocabulary(namespace="labels")
        num_labels = len(all_labels)
        constraints = allowed_transitions("BIOUL", all_labels)
        self.transition_matrix = torch.zeros([num_labels+2, num_labels+2]).fill_(0.0)

        for c in constraints:
            # if (c[0] < num_labels) and (c[1] < num_labels):
            self.transition_matrix[c[0],c[1]] = 1.0

        self.nconstraints = num_labels + 2
        
        aggregate_type = self.config.get("aggregate_type", "max")
        if aggregate_type.lower() not in ["sum", "max"]:
            raise Exception("Transition aggregation type invalid")
        if aggregate_type.lower() == "sum":
            self.aggregate_mat = lambda x: x.sum(dim=3)
        elif aggregate_type.lower() == "max":
            self.aggregate_mat = lambda x: x.max(dim=3)[0]

    def num_constraints(self):
        return self.nconstraints 

    # Pass in class PROBABILITIES, NOT logits!
    def get_penalty(self, prob0, mask, span_matrix, indicator=0):
        # prob is class PROBABILITIES, NOT logits(keep the option?)

        # prob - bsize x senlen x ntags        
        
        prob = F.pad(prob0, (0,2,1,1,0,0))
        bsize, senlen, ntags = prob.shape

        prob[:,0,ntags-2] = 1
        prob[:,-1,ntags-1] = 1

        m1 = prob[:, :-1, :]
        m2 = prob[:, 1:, :]
        
        m2 = m2.unsqueeze(-1).expand(bsize, senlen-1, ntags, ntags).transpose(2,3)
        t_mat = self.transition_matrix.unsqueeze(0).unsqueeze(0).expand(bsize, senlen-1, ntags, ntags)

        constr_mat = m2 * self.transition_matrix
        constr_mat = m1 - self.aggregate_mat(constr_mat)
        
        # Apply mask
        # Pdb().set_trace()        
        constr_mat = constr_mat * F.pad(mask, (1,0,0,0), value=1.0).unsqueeze(-1).expand_as(constr_mat)

        F.relu(constr_mat, inplace=True)
        constr_mat = constr_mat.sum(dim=1)

        return constr_mat

"""
