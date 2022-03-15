import torch
import torch.nn as nn


metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
softmax = nn.Softmax(dim=-1)
logsoftmax = nn.LogSoftmax(dim=-1)
nll = nn.CrossEntropyLoss(ignore_index=-1)
criterion_soft_entropy = lambda output, labels: -torch.mean(labels * logsoftmax(output))
squared_l2_norm = nn.MSELoss()


def cal_adj(X):
    '''
    feature matrix X: [batch, slot, dim]
    '''
    adjs = []
    bs = X.size(0)
    for s in range(X.size(1)):
        # for each slot
        featX = X[:, s, :]
        tmpX = torch.mm(featX, featX.t()) # [bs, bs]
        adj = torch.mm(tmpX.diag().view(-1, 1), torch.ones(1, bs).to(X.device)) 
        adj = adj + adj.t()
        adj = adj - 2.0 * tmpX
        adj = torch.exp(-0.5 * adj)
        
        adjs.append(adj.detach())
        
    return adjs


def slot_value_matching(hidden, value_emb):
    batch_size = hidden.size(0)
    num_slots = hidden.size(1)
    
    pred_label_distribution = []
    all_distance = []
    for s in range(num_slots): # note: slots are successive
        hidden_label = value_emb[s]
        num_slot_labels = hidden_label.size(0) # number of value choices for each slot

        _hidden_label = hidden_label.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size * num_slot_labels, -1)
        _hidden = hidden[:,s,:].unsqueeze(1).repeat(1, num_slot_labels, 1).reshape(batch_size * num_slot_labels, -1)
        _dist = metric(_hidden_label, _hidden).view(batch_size, num_slot_labels)

        _dist = -_dist

        slot_label_distribution = softmax(_dist)
        
        pred_label_distribution.append(slot_label_distribution)
        all_distance.append(_dist)

    return pred_label_distribution, all_distance


def hard_cross_entropy_loss(all_distance, labels):
    loss = 0.
    loss_slot = []
    pred_slot = []
    num_slots = len(all_distance)

    for s in range(num_slots): # note: slots are successive
        _dist = all_distance[s]
        _, pred = torch.max(_dist, -1)
        pred_slot.append(pred.view(_dist.size(0), 1))

        _loss = nll(_dist, labels[:, s])
        loss += _loss
        loss_slot.append(_loss.item())

    pred_slot = torch.cat(pred_slot, 1) # [batch_size, num_slots]

    return loss, loss_slot, pred_slot


def soft_cross_entropy_loss(all_distance, gt_distribution):
    loss = 0.
    for pred, gt in zip(all_distance, gt_distribution):
        loss += criterion_soft_entropy(pred, gt) # sum over all slots
        
    return loss


def l2_loss(preds, gts):
    loss = 0.
    num_slots = preds.size(1)
    for s in range(num_slots):
        loss += squared_l2_norm(preds[:, s, :], gts[:, s, :])
        
    return loss