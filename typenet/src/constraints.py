from collections import OrderedDict, defaultdict
import dd_utils
import torch.nn.functional as F
import torch

from penalty import BasePenalty

def get_constraints(config, aux_data):
    return Constraints(config,aux_data)



class Constraints(torch.nn.Module):
    def __init__(self, config, aux_data):
	super(Constraints, self).__init__()        
	self.constraint_dict = OrderedDict()
        
        #implication constraint
        self.constraint_dict['imp'] = Implication(config,aux_data['adj_matrix'])
        #Mutl Excl constraint:
        self.constraint_dict['me'] = MutualExclusion(config,aux_data['cooccur'])
        #Max Prob = 1 Constraint:
        self.constraint_dict['maxp'] = MaxProb1(config)
       
        self.id2key = list(self.constraint_dict.keys())
        self.key2id = dict((k,i) for i,k in enumerate(self.id2key))
        if config.dd_constant_lambda:
            self.lambda_list = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(self.constraint_dict[k].num_constraints()),requires_grad=False) for k in self.id2key])
        else:
            self.lambda_list = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.constraint_dict[k].num_constraints())) for k in self.id2key])
        #self.lambda_dict = torch.nn.ParameterDict([(k,torch.nn.Parameter(torch.zeros(v.num_constraints()))) for k,v in self.constraint_dict.items()])

        #for k, v in self.constraint_dict.items():
        #    self.lambda_dict[k] = torch.nn.Parameter(
        #        torch.zeros(v.num_constraints()))


    def get_optim_params(self,config):
        list_of_params = []
        for i,k in enumerate(self.id2key):
            factor = 1
            if config.use_wt_as_lr_factor:
                factor = self.constraint_dict[k].weight
                self.constraint_dict[k].weight = 1
                #
            if self.lambda_list[i].requires_grad:
                list_of_params.append({'params': self.lambda_list[i], 'lr': config.ddlr*factor})
        #
        return list_of_params




    def forward(self,scores):
        loss = 0.0
        penalties = {}
        for k in self.constraint_dict:
            penalty = (self.lambda_list[self.key2id[k]]*(self.constraint_dict[k].get_penalty(scores).mean(dim=0))).sum()
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


class Implication(BasePenalty):
    def __init__(self, config, adj_matrix):
        super(Implication,self).__init__()
        self.dd_pids, self.dd_cids = dd_utils.get_parent_child_tensors(
            adj_matrix)
        self.dd_logsig = config.dd_logsig
        self.dd_penalty = config.dd_penalty
        self.weight = config.dd_implication_wt 

    def num_constraints(self):
        return self.dd_pids.numel()

    def get_penalty(self, scores, indicator=0):
        # Pdb().set_trace()
        if indicator == 1:
            # scores are indicator predictions
            return F.relu(scores[:, self.dd_cids] - scores[:, self.dd_pids])
        if not self.dd_logsig:
            prob = F.sigmoid(scores)
            penalty = F.relu(prob[:, self.dd_cids] - prob[:, self.dd_pids])
        else:
            logsig = F.logsigmoid(scores)
            if self.dd_penalty == 'strict':
                penalty = F.relu(
                    logsig[:, self.dd_cids] - logsig[:, self.dd_pids])
            elif self.dd_penalty == 'relax':
                penalty = F.relu(logsig[:, self.dd_cids] - 
                        logsig[:,self.dd_pids])*((scores[:, self.dd_cids] >= 0).float())
            elif self.dd_penalty == 'mix':
                lognegsig = F.logsigmoid(-1.0*scores)
                penalty = F.relu(logsig[:, self.dd_cids] - logsig[:, self.dd_pids])*((scores[:, self.dd_cids] >= 0).float(
                )) + F.relu(lognegsig[:, self.dd_pids] - lognegsig[:, self.dd_cids])*((scores[:, self.dd_cids] < 0).float())
            else:
                raise 'incorrect penalty type'
        return penalty


class MutualExclusion(BasePenalty):
    def __init__(self, config, cooccur):
        super(MutualExclusion,self).__init__()
        self.dd_mut_excl1, self.dd_mut_excl2 = dd_utils.get_mutually_exclusive_pairs(
            cooccur)
        self.dd_logsig = config.dd_logsig
        self.weight = config.dd_mutl_excl_wt

    def num_constraints(self):
        return self.dd_mut_excl1.numel()

    def get_penalty(self, scores, indicator=0):
        if indicator == 1:
            return F.relu(scores[:, self.dd_mut_excl1] + scores[:, self.dd_mut_excl2] - 1.0)
        scores_max = torch.max(
            scores[:, self.dd_mut_excl1], scores[:, self.dd_mut_excl2])
        scores_min = torch.min(
            scores[:, self.dd_mut_excl1], scores[:, self.dd_mut_excl2])

        if not self.dd_logsig:
            penalty = F.relu(F.sigmoid(scores_max) + F.sigmoid(scores_min) - 1.0)
        else:
            penalty = F.relu(F.logsigmoid(scores_max) -
                         F.logsigmoid(-1.0*scores_min))

        return penalty

class MaxProb1(BasePenalty):
    def __init__(self,config):
        super(MaxProb1,self).__init__()
        self.dd_logsig = config.dd_logsig
        self.weight = config.dd_maxp_wt
        if hasattr(config, 'dd_maxp_thr'):
            self.thr = config.dd_maxp_thr
            assert (self.thr <=  1.0) and (self.thr > 0)
            
        else:
            self.thr = 1.0

    def num_constraints(self):
        return 1

    def get_penalty(self, scores, indicator = 0):
        if indicator == 1:
            return F.relu(1.0 - scores.max(dim=1)[0])
        else:
            if not self.dd_logsig:
                return F.relu(self.thr - F.sigmoid(scores.max(dim=1)[0]))
            else:
                return F.relu(math.log(self.thr) -1.0*F.logsigmoid(scores.max(dim=1)[0]))


