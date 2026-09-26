import os
import time
import torch
import random
import numpy as np
import pandas as pd
import argparse
import logging
import pickle
from torch import nn
import torch.nn.functional as F
import json
from torch.utils.data import Dataset, DataLoader

from models.backbone_SASRec_controlled import (
    SASRec,
    MoRec,
    UniSRec,
    RLMRec,
    LLMInit,
    AlphaFuse,
    LLMEmb,
    TedRec,
    DIFSR,
    CF_Guard,
)
from utils import evaluate

class SeqDataset(Dataset):
    def __init__(self, data, exp_type = 'main_exp', neg_data = None):
        
        self.exp_type = exp_type
        
        with open(data, "r") as f:
            lines = [list(map(int, line.strip().split())) for line in f if line.strip()]

        histories = [line[:-1] for line in lines]
        targets = [line[-1] for line in lines]

        self.max_len = 10

        self.seq_data = [
            torch.tensor([0] * (self.max_len - len(h)) + h , dtype=torch.long)
            for h in histories
        ]

        self.len_seq_data = [torch.tensor(len(h), dtype=torch.long) for h in histories]

        self.next_data = [torch.tensor(t, dtype=torch.long) for t in targets]

        if self.exp_type == 'analysis':
            
            with open(neg_data, "r") as f:
                lines = [list(map(int, line.strip().split())) for line in f if line.strip()]

            neg_data = lines
            self.negatives = [torch.tensor(t, dtype=torch.long) for t in neg_data]

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        if self.exp_type == 'main_exp':
            return {'seq': self.seq_data[idx], 'len_seq': self.len_seq_data[idx], 'next': self.next_data[idx], 'idx':idx}
        elif self.exp_type == 'analysis':
            return {'seq': self.seq_data[idx], 'len_seq': self.len_seq_data[idx], 'next': self.next_data[idx], 'neg': self.negatives[idx], 'idx':idx}
        
def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

logging.getLogger().setLevel(logging.INFO)

def setup_seed(seed): 
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")
    parser.add_argument('--random_seed', type=int, default=22)
    ### training settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--lr_delay_rate', type=float, default=0.99)
    parser.add_argument('--lr_delay_epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='games',
                        help='games, movies, baby')
    parser.add_argument('--split_mode', nargs='?', default='time',
                        help='timestamp split')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-6,
                        help='l2 loss reg coef.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    ### SASRec backbone settings & MR settings
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--num_layers', default=2, type=int)
    ### loss function parameters
    parser.add_argument('--loss_type', type=str, default="infoNCE")
    parser.add_argument('--neg_ratio', type=int, default=64,
                        help='#Negative:#Positive = neg_ratio.')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='tao.')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='scale of additional loss of RLMRec')
    ### languange embeddings settings
    parser.add_argument('--language_model_type', default="qwen3", type=str)
    parser.add_argument('--language_embs_scale', default=1, type=int)
    ### Item embeddings settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Number of hidden factors, i.e., ID embedding size.')
    parser.add_argument('--ID_embs_init_type', type=str, default="normal")
    parser.add_argument('--kept_dim', type=int, default=None,
                        help='Number of hidden factors, i.e., ID embedding size.')
    ### model selection
    parser.add_argument('--model_type', type=str, default="AlphaFuse")
    parser.add_argument('--SR_aligement_type', type=str, default="gen")
    # AlphaFuse
    parser.add_argument('--null_thres', type=float, default=None,)
    parser.add_argument('--null_dim', type=int, default=64,)
    parser.add_argument('--item_frequency_flag', type=str2bool, default=False)
    parser.add_argument('--whitening', type=str2bool, default=False)
    parser.add_argument('--cover', type=bool, default=False)
    parser.add_argument('--ID_space', type=str, default="singular")
    parser.add_argument('--inject_space', type=str, default="singular")
    parser.add_argument(
        '--controlled_mode',
        type=str,
        choices=('language', 'random', 'shuffle', 'zero'),
        default='language',
        help='Controlled AlphaFuse input language embedding variant.',
    )
    parser.add_argument('--only_eval', type=bool, default=False)
    # CF-Gurad & CFR settings
    parser.add_argument('--dropout_rate2', type=float, default=0.5,)
    parser.add_argument('--num_layers2', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=5.0,)
    parser.add_argument('--lambda_cold', type=float, default=0.5)
    parser.add_argument('--learn_dim', type=int, default=32)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--CF_model_type', type=str, default="SASRec")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    setup_seed(args.random_seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    key_words = vars(args)
    
    data_directory = f'data/{args.split_mode}/{args.data}/'
    model_directory = f'results/{args.split_mode}/{args.data}/'
    controlled_suffix = (
        f"_controlled-{args.controlled_mode}"
        if args.model_type == "AlphaFuse"
        else ""
    )

    key_words["language_embs_path"] = data_directory
    
    if args.model_type == "SASRec":
        model = SASRec(device, **key_words).to(device)  
    elif args.model_type == "MoRec": # Adaptive Projection
        model = MoRec(device, **key_words).to(device)
    elif args.model_type == "UniSRec": # Adaptive Projection
        model = UniSRec(device, **key_words).to(device)
    elif args.model_type == "LLMInit":
        model = LLMInit(device, **key_words).to(device)
    elif args.model_type == "RLMRec": # construct
        model = RLMRec(device, **key_words).to(device)
    elif args.model_type == "AlphaFuse":
        model = AlphaFuse(device, **key_words).to(device)
    elif args.model_type == "LLMEmb":
        model = LLMEmb(device, **key_words).to(device)
    elif args.model_type == "DIF-SR":
        model = DIFSR(device, **key_words).to(device) 
    elif args.model_type == "TedRec":
        model = TedRec(device, **key_words).to(device) 
    elif args.model_type == "CF-Guard":
        model = CF_Guard(device, **key_words).to(device) 
    else:
        raise NotImplementedError
        # For CCFRec, we refer to the official implementations https://github.com/BishopLiu/CCFRec

        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    print(key_words)
        
    train_data = os.path.join(data_directory, 'data_train.txt')
    #train_data.reset_index(inplace=True,drop=True)
    train_dataset = SeqDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)

    val_data = os.path.join(data_directory, 'data_valid.txt')
    #val_data.reset_index(inplace=True,drop=True)
    val_dataset = SeqDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    test_data = os.path.join(data_directory, 'data_test.txt')
    #test_data.reset_index(inplace=True,drop=True)
    test_dataset = SeqDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    best_ndcg10 = 0  
    patience = 30  
    counter = 0  
    grad_log = []
    id_log = []
    step = 0
    T = 0.0
    print("Loading Data Done.")

    ''''''
    if not args.only_eval:
        for epoch in range(args.epoch):
            model.train()
            for batch in train_loader:
                
                batch_size = len(batch['seq'])
                seq = batch['seq'].to(device)
                #len_seq = batch['len_seq'].to(device)
                target = batch['next'].to(device)
                
                optimizer.zero_grad()
                
                if args.model_type == "CF-Guard":
                    loss = model.calculate_total_loss(seq,  target, args.neg_ratio, args.temperature)
                else:
                    if args.model_type != "TedRec":
                        if args.loss_type == "CE":
                            loss = model.calculate_ce_loss(seq, target)
                        elif args.loss_type == "infoNCE":
                            loss = model.calculate_infonce_loss(seq,  target, args.neg_ratio, args.temperature)
                        
                    if args.model_type == "RLMRec":
                        recon_loss = model.reconstruct_gen_loss()
                        loss = loss + args.beta * recon_loss

                    if args.model_type == "LLMEmb":
                        loss += 1e-2 * model.pointwise_loss() 

                    if args.model_type == "TedRec":
                        loss = model.calculate_loss(seq, target, args.neg_ratio, args.temperature)
                
                loss.backward()

                optimizer.step()
                step+=1
                
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

            if (epoch+1) % 50 == 0:
                _ = evaluate(model, train_loader, device)
            if (epoch+1) % 1 == 0:
                model.eval()
                print('-------------------------- EVALUATE PHRASE --------------------------')
                t0 = time.time()
                val_ndcg10 = evaluate(model, val_loader, device)
                t1 = time.time() - t0
                print("\n using ",t1, "s ", "Eval Time Cost",T,"s.")

                model.train()
                tv_ndcg10 = val_ndcg10 
                if tv_ndcg10 > best_ndcg10:
                    best_ndcg10 = tv_ndcg10
                    counter = 0 
                    
                    print("\n best NDCG@10 is updated to ",best_ndcg10,"at epoch",epoch)
                    
                    if args.model_type != "CF-Guard":
                        epoch_str = f"SASRec_{args.model_type}_rs{args.random_seed}_IDdim{args.hidden_dim}_Textdim{args.null_dim}_{args.lr}_{args.loss_type}_{args.ID_embs_init_type}{controlled_suffix}.pth"
                    else:
                        epoch_str = f"{args.CF_model_type}_SASRec_{args.model_type}_rs{args.random_seed}_IDdim{args.hidden_dim}_Textdim{args.null_dim}_{args.lr}_{args.loss_type}_{args.ID_embs_init_type}.pth"

                    os.makedirs(model_directory, exist_ok=True)
                    torch.save(model.state_dict(), model_directory + epoch_str)
                else:
                    counter += 1
                    if counter >= patience:
                        break   
                print('----------------------------------------------------------------')

    if args.model_type != "CF-Guard":
        epoch_str = f"SASRec_{args.model_type}_rs{args.random_seed}_IDdim{args.hidden_dim}_Textdim{args.null_dim}_{args.lr}_{args.loss_type}_{args.ID_embs_init_type}{controlled_suffix}.pth"
    else:
        epoch_str = f"{args.CF_model_type}_SASRec_{args.model_type}_rs{args.random_seed}_IDdim{args.hidden_dim}_Textdim{args.null_dim}_{args.lr}_{args.loss_type}_{args.ID_embs_init_type}.pth"
    
    model.load_state_dict(torch.load(model_directory + epoch_str))

    model.eval()
    
    print('-------------------------- TEST RESULTS --------------------------')
    
    _ = evaluate(model, test_loader, device)
    
    print("Done.")


    
