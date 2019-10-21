
import torch.nn.functional as F
import torch
import numpy as np

def get_parent(ind,x):
    nonzero = np.where(x)[0]
    assert len(nonzero) <= 2 and len(nonzero) >= 1
    if len(nonzero) == 1:
        return  nonzero[0]
    else:
        return nonzero[nonzero != ind][0]

def get_parent_child_tensors(typenet_matrix):
    parent_ids = []
    child_ids = []
    for child_id in range(typenet_matrix.shape[0]):
        for parent_id in np.where(typenet_matrix[child_id] == 1)[0]:
            if parent_id != child_id:
                parent_ids.append(parent_id)
                child_ids.append(child_id)

    return (torch.LongTensor(parent_ids), torch.LongTensor(child_ids))



def get_mutually_exclusive_pairs(cooccur_matrix):
    type1,type2 = np.where(cooccur_matrix == 0)
    ind = type1 > type2
    type1  = type1[ind]
    type2 = type2[ind]
    return (torch.LongTensor(type1), torch.LongTensor(type2))


def get_implication_penalty(scores,dd_pids, dd_cids, config, indicator = 0):
    #Pdb().set_trace()
    if indicator == 1:
        #scores are indicator predictions
        return F.relu(scores[:,dd_cids] - scores[:,dd_pids])
    if not config.dd_logsig:
        prob = F.sigmoid(scores)
        penalty = F.relu(prob[:,dd_cids] - prob[:,dd_pids])
    else:
        logsig  = F.logsigmoid(scores)
        if config.dd_penalty == 'strict':
            penalty = F.relu(logsig[:,dd_cids] - logsig[:,dd_pids])
        elif config.dd_penalty == 'relax':
            penalty = F.relu(logsig[:,dd_cids] - logsig[:,dd_pids])*((scores[:,dd_cids] >= 0).float())
        elif config.dd_penalty == 'mix':
            lognegsig = F.logsigmoid(-1.0*scores)
            penalty =  F.relu(logsig[:,dd_cids] - logsig[:,dd_pids])*((scores[:,dd_cids] >= 0).float()) +  F.relu(lognegsig[:,dd_pids] - lognegsig[:,dd_cids])*((scores[:,dd_cids] < 0).float())
        else: 
            raise 'incorrect penalty type'
    return penalty

def get_mut_excl_penalty(scores, dd_mut_excl1, dd_mut_excl2, config,indicator = 0):
    if indicator == 1:
        return F.relu(scores[:,dd_mut_excl1] + scores[:,dd_mut_excl2] - 1.0)
    scores_max = torch.max(scores[:,dd_mut_excl1], scores[:,dd_mut_excl2])
    scores_min = torch.min(scores[:,dd_mut_excl1], scores[:,dd_mut_excl2]) 
    
    if not config.dd_logsig:
        penalty = F.relu(F.sigmoid(scores_max)  +  F.sigmoid(scores_min) - 1.0)
    else:
        penalty  = F.relu(F.logsigmoid(scores_max) - F.logsigmoid(-1.0*scores_min))
    
    return penalty 


def get_cooccur(data):
    max_id  = 0
    for x in data:
        max_id = max(max_id,max(x))

    cooccur = np.zeros((max_id+1,max_id+1))

    for x in data:
        for i in x:
            for j in x:
                if i != j:
                    cooccur[i,j] += 1
    return cooccur





def get_parent_index(typenet_matrix):
    return(torch.tensor([get_parent(i,x) for (i,x) in  enumerate(typenet_matrix)]).long())
