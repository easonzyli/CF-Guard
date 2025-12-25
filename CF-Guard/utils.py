import os
import copy
import math
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn
import torch
import torch.nn.functional as F
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def left_pad_sequence(sequences, batch_first=True, padding_value=0):

    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    
    for seq in sequences:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            pad = torch.full((pad_len,), padding_value, dtype=seq.dtype, device=seq.device)
            padded_seq = torch.cat([pad, seq], dim=0)  # pad on the LEFT
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    
    padded_batch = torch.stack(padded_seqs, dim=0)  # (B, L)
    if not batch_first:
        padded_batch = padded_batch.transpose(0, 1)  # (L, B)
    return padded_batch


class Collator(object):
    def __init__(self, ):
        self.n_digits = 4
        self.max_his_len = 10

    def __call__(self, batch):
        item_inters, targets, code_inters, mask_targets = zip(*batch)
        
        item_inters = [torch.tensor(x, dtype=torch.long) for x in item_inters]
        item_inters = left_pad_sequence(item_inters, batch_first=True, padding_value=0)  # (B, L)

        code_inters = [torch.tensor(x, dtype=torch.long) for x in code_inters]
        code_inters = left_pad_sequence(code_inters, batch_first=True, padding_value=0)  # (B, L, ?)

        max_len = item_inters.size(1)
        padded_code_inters = []
        for code_seq in code_inters:
            pad_len = max_len - code_seq.size(0)
            if pad_len > 0:
                pad = torch.full((pad_len, self.n_digits), 0, dtype=torch.long)
                padded_code = torch.cat([pad, code_seq], dim=0)  # left pad
            else:
                padded_code = code_seq
            padded_code_inters.append(padded_code)
        code_inters = torch.stack(padded_code_inters, dim=0)  # (B, L, n_digits)
        code_inters = code_inters.reshape(-1, self.n_digits)

        targets = torch.tensor(targets, dtype=torch.long)

        mask_targets = [torch.tensor(x, dtype=torch.long) for x in mask_targets]
        mask_targets = left_pad_sequence(mask_targets, batch_first=True, padding_value=-100)
        mask_targets = mask_targets.reshape(-1)

        seq_lens = torch.tensor([len(x) for x in item_inters], dtype=torch.long)


        return dict(seq=item_inters.long(),
                    code_inters_a=code_inters.long(),
                    inter_lens=seq_lens.long(),
                    next=targets.long(),
                    mask_targets_a=mask_targets.long())



def random_neq(l, r, s=[]):   
    
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def calculate_hit_loader(sorted_list,topk,true_items,hit_purchase,ndcg_purchase,mrr_purchase):
    true_items = true_items.tolist()
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        # print(rec_list)
        # print(true_items)
        # print('...........')
        # break
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])[0,0]
                # total_reward[i] += rewards[j]
                # if rewards[j] == r_click:
                #     hit_click[i] += 1.0
                #     ndcg_click[i] += 1.0 / np.log2(rank + 1)
                # else:
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
                mrr_purchase[i] += 1.0/ rank


def evaluate(model, val_loader, device):

    total_purchase = 0.0
    hit_purchase=[0,0,0,0,0]
    ndcg_purchase=[0,0,0,0,0]
    mrr_purchase = [0,0,0,0,0]
    topk = [1,5,10,20,50]
    sample_records = []
    for batch in val_loader:
        seq = batch["seq"].to(device)
        #idxs = batch["idx"]
        #len_seq = batch["len_seq"]
        target = batch["next"]
        #idxs = batch["idx"].tolist()
        #labels = batch['labels']
        #prediction = model.predict(seq,len_seq,target, emb_type)
        
        #neg = batch['neg']
        
        prediction = model.predict(seq)
        
        #print(prediction.shape)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        total_purchase+=len(seq)
        
    
    hr_list = []
    ndcg_list = []
    mrr_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        mr_purchase=mrr_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    return ndcg_list[2]

def evaluate_bert(model, val_loader, device):

    total_purchase = 0.0
    hit_purchase=[0,0,0,0,0]
    ndcg_purchase=[0,0,0,0,0]
    mrr_purchase = [0,0,0,0,0]
    topk = [1,5,10,20,50]
    sample_records = []
    for batch in val_loader:
        seq = batch["seq"].to(device)
        len_seq = batch["len_seq"]
        target = batch["next"]
        idxs = batch["idx"].tolist()
        labels = batch['labels']
        positions = batch['position']
        prediction = model.predict(seq,len_seq,target)

        #prediction = model.predict(seq, positions)
        #print(prediction.shape)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        total_purchase+=len(seq)
        
    
    
    hr_list = []
    ndcg_list = []
    mrr_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        mr_purchase=mrr_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    return ndcg_list[2]