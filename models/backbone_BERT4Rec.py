import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *
import torch
import torch.nn.functional as F
import pickle
from collections import defaultdict
import json

class ContrastiveLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x, y, gathered = False):
        
        all_y = y
        x = F.normalize(x, dim=-1)
        all_y = F.normalize(all_y, dim=-1)
        
        B = x.shape[0]
        
        logits = torch.matmul(x, all_y.transpose(0, 1)) / self.tau
        labels = torch.arange(B, device=x.device, dtype=torch.long)
        
        
        loss = F.cross_entropy(logits, labels)

        return loss


class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, **key_words):
        super(Item_Embedding, self).__init__()
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.state_size = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        self.padding_value = 0
        s_p = key_words["split_mode"]
        data = key_words["data"]
        self.pretrained_dim = key_words["learn_dim"]
        self.pop_path = f"data/{s_p}/{data}/item_freq.json"
        self.tau = key_words["temp"]
        with open(self.pop_path, "r") as f:
            self.freq_data = json.load(f)
           #pd.read_pickle(os.path.join(directory, f'{language_model_type}_embeddings.pkl'))
        self.item_freq = defaultdict(int) 
        
        for k, v in self.freq_data.items():
            self.item_freq[int(k)] = v
        
        self.item_freq[self.padding_value] = 0
        
        self.construct_item_embeddings(emb_pipline, **key_words)
    
    def construct_item_embeddings(self, emb_pipline, **key_words):
        
        if emb_pipline == "ID":
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
        
        elif emb_pipline == "SI": # LLMInit
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
            
        elif emb_pipline == "SR": # RLMRec
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([padding_emb, language_embs])
            
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.padding_value,
            )

        elif emb_pipline == "TD": # TedRec
            
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([padding_emb, language_embs])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.padding_value
                )
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
        
        elif emb_pipline == "DIF": # DIF-SR
            
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([padding_emb, language_embs])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.padding_value
                )
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)

        elif emb_pipline == "LE": # LLMEmb
            
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)

            language_embs = self.semantic_space_decomposion( key_words["hidden_dim"], **key_words)
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([padding_emb, language_embs])
            
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.padding_value
                )
            
            enhanced_cf_embs = self.load_cf_embeddings(key_words["language_embs_path"])
            self.enhanced_cf_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(enhanced_cf_embs,dtype=torch.float32),
            freeze=True,
            padding_idx=self.padding_value
            )

            
        elif emb_pipline == "AP": # MoRec & UniSRec

            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([ padding_emb, language_embs])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.padding_value
                )

            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
            
        elif emb_pipline == "AF": # AlphaFuse
            key_words["item_frequency_flag"] = False
            key_words['whitening'] = True
            
            self.cliped_language_embs = self.semantic_space_decomposion( key_words["hidden_dim"],  **key_words)

            padding_emb = np.random.rand(self.cliped_language_embs.shape[1])  # padding ID embedding
            self.cliped_language_embs_with_padding = np.vstack([padding_emb, self.cliped_language_embs])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(self.cliped_language_embs_with_padding, dtype=torch.float32),
                freeze=True,
                padding_idx=self.padding_value
                )
            self.init_ID_embedding(self.nullity, key_words["ID_embs_init_type"], **key_words)


        elif emb_pipline == "CFG": # CF-Guard
            key_words["item_frequency_flag"] = False
            key_words['whitening'] = True
            print(key_words['whitening'])
            item_ids = sorted(self.item_freq.keys()) 
            item_freq_array = np.array([self.item_freq[i] for i in item_ids])

            threshold_value = np.percentile(item_freq_array, 100 * key_words["lambda_cold"])
            print(f"threshold_value: {threshold_value}")
            hot_mask_np = item_freq_array >= threshold_value
            cold_mask_np = ~hot_mask_np
            
            self.hot_mask = torch.tensor(hot_mask_np, dtype=torch.bool)
            self.cold_mask = torch.tensor(cold_mask_np, dtype=torch.bool, device="cuda")
            self.hot_items = torch.nonzero(self.hot_mask, as_tuple=True)[0]
            self.cold_items = torch.nonzero(self.cold_mask, as_tuple=True)[0]
            
            self.cliped_language_embs = self.semantic_space_decomposion( key_words["hidden_dim"],  **key_words)
            
            padding_emb = np.random.rand(self.cliped_language_embs.shape[1])  # padding ID embedding
            self.cliped_language_embs_with_padding = np.vstack([padding_emb, self.cliped_language_embs])
            
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(self.cliped_language_embs_with_padding, dtype=torch.float32),
                freeze=True,
                padding_idx=self.padding_value
                )
            
            language_embs_tensor = torch.tensor(self.cliped_language_embs_with_padding, dtype=torch.float32)  # [num_items, dim]

            language_embs_tensor = F.normalize(language_embs_tensor, p=2, dim=-1)  # [num_items, dim]

            self.text_similarity = language_embs_tensor @ language_embs_tensor.T
            self.text_similarity = self.text_similarity.fill_diagonal_(-1e1).to("cuda").detach()
            self.text_similarity = F.softmax(self.text_similarity / self.tau , dim=-1)  # [num_items, num_items]

    def load_language_embeddings(self, directory, language_model_type, scale = 1.0):
        language_embs = pd.read_pickle(os.path.join(directory, f'{language_model_type}_embeddings.pkl'))
        self.item_num = len(language_embs)
        self.language_dim = len(language_embs[0])
        return np.stack(language_embs) 
    
    def load_cf_embeddings(self, directory, pretrain_cf_drop = 0.1, dim = 128):
        print(f"Load BERT4Rec_qwen3_normal_0.001_{pretrain_cf_drop}_{dim}.pkl")
        cf_embs = pd.read_pickle(os.path.join(directory, f'BERT4Rec_qwen3_normal_0.001_{pretrain_cf_drop}_{dim}.pkl'))

        self.cf_dim = len(cf_embs[0])
        return np.stack(cf_embs) 


    def init_ID_embedding(self, ID_dim, init_type, **key_words):
        
        if init_type == "language_embeddings":
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            if self.language_dim == ID_dim:
                padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
                language_embs = np.vstack([padding_emb, language_embs])
                #language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.padding_value
                    )
            else:
                clipped_language_embs = self.semantic_space_decomposion(ID_dim,  **key_words)
                padding_emb = np.random.rand(clipped_language_embs.shape[1])  # padding ID embedding
                clipped_language_embs = np.vstack([padding_emb, clipped_language_embs])
                #language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.padding_value
                    )
        else:
            print("Random_Init_ID")
            if "CF-Guard" in key_words["model_type"]:
                print("CF_Net init!")
                self.ID_embeddings = nn.Embedding(
                    num_embeddings=self.item_num+1,
                    embedding_dim=key_words["learn_dim"],
                )
            else:
                self.ID_embeddings = nn.Embedding(
                    num_embeddings=self.item_num+1,
                    embedding_dim=ID_dim,
                )
            
            nn.init.normal_(self.ID_embeddings.weight, 0, 1)
                   
    def semantic_space_decomposion(self, clipped_dim, **key_words):
        language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])

        # The default item distribution is a uniform distribution.
        self.language_mean = np.mean(language_embs, axis=0)
        cov = np.cov( language_embs - self.language_mean, rowvar=False)
        
            #raise NotImplementedError("Custom item distribution is not implemented yet.")
        U, S, _ = np.linalg.svd(cov, full_matrices=False)
        
        if key_words["null_thres"] is not None:
            indices_null = np.where(S <= key_words["null_thres"])[0]
            self.nullity = len(indices_null)
        elif key_words["null_dim"] is not None:
            self.nullity = key_words["null_dim"]
        #print("The Nullity is", self.nullity)
        #self.squared_singular_values = S
        #self.language_bases = U
        if clipped_dim is None:
            clipped_dim = self.language_dim
        if key_words["cover"]:
            clipped_dim = clipped_dim - self.nullity
        
        Projection_matrix = U[...,:clipped_dim]
        
        if key_words['whitening']:
            print("Whitening!")
            Diagnals = np.sqrt(1/S)[:clipped_dim]
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals)) # V_{\lamda} into V_1"""
        
        clipped_language_embs = (language_embs-self.language_mean).dot(Projection_matrix)


        return clipped_language_embs

class CF_backbone(nn.Module):
    def __init__(self, device, **key_words):
        super(CF_backbone, self).__init__()
        
        self.cf_enhanced = key_words["cf_enhanced"]
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        self.padding_value = 0
        #self.item_embeddings = Item_Embedding("ID", **key_words)
        #self.item_num = item_num
        #self.seq_len = seq_len
        self.num_layers = int(key_words["num_layers2"])
        self.dropout = key_words["dropout_rate2"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        #self.language_dim = self.item_embeddings.language_dim
        self.hidden_dim = key_words["learn_dim"]

        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.hidden_dim
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.mh_attn = Bid_MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
        #self.s_fc = nn.Linear(self.hidden_size, self.item_num)
        # self.ac_func = nn.ReLU()
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_1": nn.LayerNorm(self.hidden_dim),
                "ln_2": nn.LayerNorm(self.hidden_dim),
                "ln_3": nn.LayerNorm(self.hidden_dim),
                "mh_attn": Bid_MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout),
                "ff": PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
            })
            for _ in range(self.num_layers)
        ])

    def embed_ID(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        pass

    def save_item_embeddings(self, path):
        """保存 item embeddings 到文件"""
        item_embs = self.return_item_emb().detach().cpu().numpy()
        with open(path, "wb") as f:
            pickle.dump(item_embs, f)
        print(f"[AlphaFuse] Item embeddings saved to {path}")
    
    def return_item_emb(self,):
        #return self.item_embeddings.ID_embeddings.weight 
        pass
    
    def forward(self, sequences):
        inputs_emb = self.embed_ID(sequences)
        #print("seq_len:", self.seq_len)
        #print("positional_embeddings size:", self.positional_embeddings.weight.shape)
        inputs_emb += self.positional_embeddings(torch.arange(inputs_emb.shape[1]).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.padding_value).float().unsqueeze(-1).to(self.device)
        seq *= mask
        
        x = seq
        for layer in self.encoder_layers:
            attn_out = layer["mh_attn"](layer["ln_1"](x), x)
            ff_out = layer["ff"](layer["ln_2"](attn_out))
            x = ff_out * mask

            x = layer["ln_3"](x)

        logits = x[:, -1]  # [B, H]
        return logits
    
class CF_backbone(nn.Module):
    def __init__(self, device, **key_words):
        super(CF_backbone, self).__init__()
        
        self.cf_enhanced = key_words["cf_enhanced"]
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        self.padding_value = 0
        #self.item_embeddings = Item_Embedding("ID", **key_words)
        #self.item_num = item_num
        #self.seq_len = seq_len
        
        self.dropout = key_words["dropout_rate2"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        #self.language_dim = self.item_embeddings.language_dim
        self.hidden_dim = key_words["hidden_dim"]
        self.num_layers = int(key_words["num_layers2"])
        
        self.hidden_dim = key_words["learn_dim"]
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=key_words["learn_dim"]
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.mh_attn = Cau_MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
        #self.s_fc = nn.Linear(self.hidden_size, self.item_num)
        # self.ac_func = nn.ReLU()
        
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_1": nn.LayerNorm(self.hidden_dim),
                "ln_2": nn.LayerNorm(self.hidden_dim),
                "ln_3": nn.LayerNorm(self.hidden_dim),
                "mh_attn": Cau_MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout),
                "ff": PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
            })
            for _ in range(self.num_layers)
        ])

    def embed_ID(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        pass

    def save_item_embeddings(self, path):
        """保存 item embeddings 到文件"""
        item_embs = self.return_item_emb().detach().cpu().numpy()
        with open(path, "wb") as f:
            pickle.dump(item_embs, f)
        print(f"[AlphaFuse] Item embeddings saved to {path}")
    
    def return_item_emb(self,):
        #return self.item_embeddings.ID_embeddings.weight 
        pass
    
    def forward(self, sequences):
        inputs_emb = self.embed_ID(sequences)
        #print("seq_len:", self.seq_len)
        #print("positional_embeddings size:", self.positional_embeddings.weight.shape)
        inputs_emb += self.positional_embeddings(torch.arange(inputs_emb.shape[1]).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.padding_value).float().unsqueeze(-1).to(self.device)
        seq *= mask
        
        x = seq
        for layer in self.encoder_layers:
            attn_out = layer["mh_attn"](layer["ln_1"](x), x)
            ff_out = layer["ff"](layer["ln_2"](attn_out))
            x = ff_out * mask

            x = layer["ln_3"](x)

        logits = x[:, -1]  # [B, H]
        return logits


class CF_Bert_backbone(nn.Module):
    def __init__(self, device, **key_words):
        super(CF_Bert_backbone, self).__init__()
        
        self.cf_enhanced = key_words["cf_enhanced"]
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        self.padding_value = 0
        #self.item_embeddings = Item_Embedding("ID", **key_words)
        #self.item_num = item_num
        #self.seq_len = seq_len
        
        self.dropout = key_words["dropout_rate2"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        #self.language_dim = self.item_embeddings.language_dim
        self.hidden_dim = key_words["hidden_dim"]
        self.num_layers = int(key_words["num_layers2"])
        
        self.hidden_dim = key_words["learn_dim"]
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=key_words["learn_dim"]
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.mh_attn = Bid_MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
        #self.s_fc = nn.Linear(self.hidden_size, self.item_num)
        # self.ac_func = nn.ReLU()
        
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_1": nn.LayerNorm(self.hidden_dim),
                "ln_2": nn.LayerNorm(self.hidden_dim),
                "ln_3": nn.LayerNorm(self.hidden_dim),
                "mh_attn": Bid_MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout),
                "ff": PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
            })
            for _ in range(self.num_layers)
        ])

    def embed_ID(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        pass

    def save_item_embeddings(self, path):
        """保存 item embeddings 到文件"""
        item_embs = self.return_item_emb().detach().cpu().numpy()
        with open(path, "wb") as f:
            pickle.dump(item_embs, f)
        print(f"[AlphaFuse] Item embeddings saved to {path}")
    
    def return_item_emb(self,):
        #return self.item_embeddings.ID_embeddings.weight 
        pass
    
    def forward(self, sequences):
        inputs_emb = self.embed_ID(sequences)
        #print("seq_len:", self.seq_len)
        #print("positional_embeddings size:", self.positional_embeddings.weight.shape)
        inputs_emb += self.positional_embeddings(torch.arange(inputs_emb.shape[1]).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.padding_value).float().unsqueeze(-1).to(self.device)
        seq *= mask
        
        x = seq
        for layer in self.encoder_layers:
            attn_out = layer["mh_attn"](layer["ln_1"](x), x)
            ff_out = layer["ff"](layer["ln_2"](attn_out))
            x = ff_out * mask

            x = layer["ln_3"](x)

        logits = x[:, -1]  # [B, H]
        return logits


class CF_GRU_backbone(nn.Module):
    def __init__(self, device, **key_words):
        super(CF_GRU_backbone, self).__init__()
        
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        #self.load_item_embeddings(pretrained_item_embeddings)
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]

        self.device = device

        self.gru_layers = nn.GRU(
            input_size=key_words['learn_dim'],
            hidden_size=key_words['learn_dim'],
            num_layers=key_words['num_layers2'],
            bias=False,
            batch_first=True,
        )
        self.padding_value = 0
        self.emb_dropout = nn.Dropout(key_words['dropout_rate2'])

        # Initialize loss function
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.hidden_dim = key_words["hidden_dim"]

    def embed_ID(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        pass

    def save_item_embeddings(self, path):
        """保存 item embeddings 到文件"""
        item_embs = self.return_item_emb().detach().cpu().numpy()
        with open(path, "wb") as f:
            pickle.dump(item_embs, f)
        print(f"[AlphaFuse] Item embeddings saved to {path}")
    
    def return_item_emb(self,):
        #return self.item_embeddings.ID_embeddings.weight 
        pass
    
    def forward(self, sequences):
        inputs_emb = self.embed_ID(sequences)
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.padding_value).float().unsqueeze(-1).to(inputs_emb.device)
        seq = seq * mask
        seq, _ = self.gru_layers(seq)
        state_hidden = seq[:, -1, :]  #extract_axis_1(seq, sequences - 1).squeeze()
        return state_hidden

class CFNet_Bert(CF_Bert_backbone):
    def __init__(self, device, text_similary, cold_items, **key_words):
        super().__init__(device, **key_words)
        self.use_eco = key_words["hot_threshold"] > 0
        self.item_embeddings = Item_Embedding("ID", **key_words)
        self.text_similarity = text_similary
        self.cold_items = cold_items
        self.lamda = key_words["lamda"]

    def embed_ID(self, x):
        #print(x)
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 

    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        

        batch_size = target.shape[0]
        neg_samples = torch.randint(1, self.item_num+1, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()

        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)

        log_feats = self.forward(sequences)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)

        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)
        
        if self.use_eco:
            cold_embs = self.embed_ID(self.cold_items)   # [cold_items, d]
            cold_embs = F.normalize(cold_embs, p=2, dim=-1)
            cold_logits = torch.matmul(log_feats, cold_embs.T)  # [B, num_cold_items]
            
            similarity = self.text_similarity[target][:, self.cold_items] # (B, num_cold_items)       
            
            cold_probs = cold_logits
            eco_loss =  - (similarity * cold_probs).sum(dim=-1) .mean()

            loss = loss + self.lamda * eco_loss

        return loss


class CFNet_GRU(CF_GRU_backbone):
    def __init__(self, device, text_similary, cold_items, **key_words):
        super().__init__(device, **key_words)
        self.use_eco = key_words["hot_threshold"] > 0
        self.item_embeddings = Item_Embedding("ID", **key_words)
        self.text_similarity = text_similary
        self.cold_items = cold_items
        self.lamda = key_words["lamda"]

    def embed_ID(self, x):

        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 

    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        
        batch_size = target.shape[0]
        neg_samples = torch.randint(1, self.item_num+1, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()

        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        #pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)

        log_feats = self.forward(sequences)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)

        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)
        
        if self.use_eco:
            cold_embs = self.embed_ID(self.cold_items)   # [cold_items, d]
            cold_embs = F.normalize(cold_embs, p=2, dim=-1)
            cold_logits = torch.matmul(log_feats, cold_embs.T)  # [B, num_cold_items]
            
            similarity = self.text_similarity[target][:, self.cold_items] # (B, num_cold_items)       
            
            cold_probs = cold_logits
            eco_loss =  - (similarity * cold_probs).sum(dim=-1) .mean()

            loss = loss + self.lamda * eco_loss

        return loss



class CFNet(CF_backbone):
    def __init__(self, device, text_similary, cold_items, **key_words):
        super().__init__(device, **key_words)
        self.use_eco = key_words["hot_threshold"] >= 0 and key_words["hot_threshold"] < 1.0
        self.item_embeddings = Item_Embedding("ID", **key_words)
        self.text_similarity = text_similary
        self.cold_items = cold_items
        self.lamda = key_words["lamda"]

    def embed_ID(self, x):
        #print(x)
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 

    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        
        batch_size = target.shape[0]
        neg_samples = torch.randint(1, self.item_num+1, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)

        log_feats = self.forward(sequences)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)

        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)
        
        if self.use_eco:
            cold_embs = self.embed_ID(self.cold_items)   # [cold_items, d]
            cold_embs = F.normalize(cold_embs, p=2, dim=-1)
            cold_logits = torch.matmul(log_feats, cold_embs.T)  # [B, num_cold_items]
            
            similarity = self.text_similarity[target][:, self.cold_items] # (B, num_cold_items)       
            
            cold_probs = cold_logits
            eco_loss =  - (similarity * cold_probs).sum(dim=-1) .mean()

            loss = loss + self.lamda * eco_loss

        return loss


class Bert4Rec_backbone(nn.Module):
    def __init__(self, device, **key_words):
        super(Bert4Rec_backbone, self).__init__()
        
        self.cf_enhanced = key_words["cf_enhanced"]
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        self.padding_value = 0

        self.num_layers = key_words["num_layers"]
        self.dropout = key_words["dropout_rate"]
        self.device = device
        self.hidden_size = key_words["hidden_dim"]
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.hidden_dim = key_words["hidden_dim"]

        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.hidden_dim
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.mh_attn = Bid_MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)
        self.ac_func = nn.ReLU()
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_1": nn.LayerNorm(self.hidden_dim),
                "ln_2": nn.LayerNorm(self.hidden_dim),
                "ln_3": nn.LayerNorm(self.hidden_dim),
                "mh_attn": Bid_MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout),
                "ff": PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
            })
            for _ in range(self.num_layers)
        ])

    def embed_ID(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        pass

    def save_item_embeddings(self, path):
        item_embs = self.return_item_emb().detach().cpu().numpy()
        with open(path, "wb") as f:
            pickle.dump(item_embs, f)
        print(f"Item embeddings saved to {path}")
    
    def return_item_emb(self,):
        #return self.item_embeddings.ID_embeddings.weight 
        pass
    
    def forward(self, sequences):
        inputs_emb = self.embed_ID(sequences)

        inputs_emb += self.positional_embeddings(torch.arange(inputs_emb.shape[1]).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.padding_value).float().unsqueeze(-1).to(self.device)
        seq *= mask
        
        x = seq
        for layer in self.encoder_layers:
            attn_out = layer["mh_attn"](layer["ln_1"](x), x)
            ff_out = layer["ff"](layer["ln_2"](attn_out))
            x = ff_out * mask

            x = layer["ln_3"](x)

        logits = x[:, -1]  # [B, H]
        return logits

    
    def predict(self, sequences):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        state_hidden = self.forward(sequences)
        item_embs = self.return_item_emb() 
        scores = torch.matmul(state_hidden, item_embs.transpose(0, 1))  # (B,|I|)
        return scores
    
    def predict_neg(self, sequences, negatives, targets): # return (B, top-100)
        candidates = torch.cat([targets.unsqueeze(1), negatives], dim=1) # (B, N+1)
        state_hidden = self.forward(sequences)
        state_hidden = state_hidden.unsqueeze(1)

        item_embs = self.return_item_emb() 
        cand_embs = item_embs[candidates]
        scores = torch.bmm(state_hidden, cand_embs.transpose(1, 2))  # (B, 1, M)
        scores = scores.squeeze(1)  # (B, M)
        return scores
    

    def calculate_ce_loss(self, sequences, target):
        seq_output = self.forward(sequences)
        item_embs = self.return_item_emb() # (|I|,d)

        logits = torch.matmul(seq_output, item_embs[1:].transpose(0, 1))
        loss = self.ce_loss(logits, target)
        return loss
    
    def calculate_bce_loss(self, sequences, target, neg_ratio, emb_type="both"):
        
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()

        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target

        target_neg = neg_samples.to(target.device)

        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        
        log_feats = self.forward(sequences)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats.unsqueeze(1) * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
        loss = self.bce_loss(pos_logits, pos_labels)
        loss += self.bce_loss(neg_logits, neg_labels)

        return loss
    
    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        
        batch_size = target.shape[0]
        neg_samples = torch.randint(1, self.item_num+1, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        #pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        log_feats = self.forward(sequences)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)
        
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)
        return loss

### pure ID embeddings
class Bert4Rec(Bert4Rec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("ID", **key_words)
 
    def embed_ID(self, x):
        #print(x)
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 
    
class MoRec(Bert4Rec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("AP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.cf_enhanced_method = key_words["cf_enhanced_method"]
        
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, key_words['hidden_dim']),
            nn.GELU()
        )

    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x) # 
        cf_embs = self.item_embeddings.ID_embeddings(x)
        return self.adapter(language_embs) + cf_embs
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        cf_embs = self.item_embeddings.ID_embeddings.weight

        return self.adapter(language_embs ) + cf_embs

class LLMEmb(Bert4Rec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("LE", **key_words)
        self.language_dim = self.item_embeddings.language_dim

        self.adapter = nn.Sequential(
            nn.Linear(key_words['hidden_dim'], key_words['hidden_dim'],bias=True),
        )
        self.log_flag = 0
        self.cold_threshold = key_words["cold_threshold"]

    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x) # 
        cf_embs = self.item_embeddings.ID_embeddings(x)
        return self.adapter(language_embs) + cf_embs
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        cf_embs = self.item_embeddings.ID_embeddings.weight

        return self.adapter(language_embs ) + cf_embs

    def pointwise_loss(self, ):
        
        language_embs = self.return_item_emb() #[n+1, d_llm]
        ID_embs = self.item_embeddings.enhanced_cf_embeddings.weight
        language_embs = F.normalize(language_embs, p=2, dim=-1)
        ID_embs = F.normalize(ID_embs, p=2, dim=-1)
        return 1 - (language_embs * ID_embs).sum() / self.item_num

class LLMInit(Bert4Rec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("SI", **key_words)
        #self.language_dim = self.item_embeddings.language_dim
 
    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 


class TedRec(Bert4Rec_backbone):
    """Text-ID fusion approach for sequential recommendation
    """
    def __init__(self, device,  **key_words):
        super().__init__(device,  **key_words)

        self.temperature = key_words['temperature']
        self.item_embeddings = Item_Embedding("TD", **key_words)
        self.hidden_size = key_words["hidden_dim"]
        self.language_dim = self.item_embeddings.language_dim
        self.item_gating = nn.Linear(self.hidden_size, 1)
        self.fusion_gating = nn.Linear(self.hidden_size, 1)
        self.plm_embedding = self.item_embeddings.language_embeddings
        self.loss_fct = nn.CrossEntropyLoss()
        self.moe_adaptor = MoEAdaptorLayer(
            8,
            [self.language_dim, key_words['hidden_dim']],
            0.2
        )

        self.complex_weight = nn.Parameter(torch.randn(1, self.seq_len // 2 + 1, key_words['hidden_dim'], 2, dtype=torch.float32) * 0.02)

        self.item_gating.weight.data.normal_(mean = 0, std = 0.02)
        self.fusion_gating.weight.data.normal_(mean = 0, std = 0.02)
        
    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion
        """
        feature_fft = torch.fft.rfft(feature_emb, dim=1, norm='ortho')
        item_fft = torch.fft.rfft(item_emb, dim=1, norm='ortho')

        complext_weight = torch.view_as_complex(self.complex_weight)
        item_conv = torch.fft.irfft(item_fft * complext_weight, n = feature_emb.shape[1], dim = 1, norm = 'ortho')
        fusion_conv = torch.fft.irfft(feature_fft * item_fft, n = feature_emb.shape[1], dim = 1, norm = 'ortho')

        item_gate_w = self.item_gating(item_conv)
        fusion_gate_w = self.fusion_gating(fusion_conv)

        contextual_emb = 2 * (item_conv * torch.sigmoid(item_gate_w) + fusion_conv * torch.sigmoid(fusion_gate_w))
        return contextual_emb

    def forward(self, item_seq, seq_text_embs, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.positional_embeddings(position_ids)
        
        input_emb = self.contextual_convolution(self.item_embeddings.ID_embeddings(item_seq), seq_text_embs )
        input_emb = input_emb + position_embedding
        
        input_emb = self.emb_dropout(input_emb)

        mask = torch.ne(item_seq, self.padding_value).float().unsqueeze(-1).to(self.device)
        input_emb *= mask
        x = input_emb
        for layer in self.encoder_layers:
            attn_out = layer["mh_attn"](layer["ln_1"](x), x)
            ff_out = layer["ff"](layer["ln_2"](attn_out))
            x = ff_out * mask

            x = layer["ln_3"](x)
        logits = x[:, -1]  # [B, H]
        return logits  

    def calculate_loss(self, sequences, target, neg_ratio, temperature):
        
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        #pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        
        item_seq = sequences #[self.ITEM_SEQ]
        #item_seq_len = interaction#[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))

        log_feats = self.forward(item_seq, item_emb_list, self.seq_len)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)

        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)
        
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def predict(self, interaction):
        item_seq = interaction#[self.ITEM_SEQ]

        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, self.seq_len)
        test_items_emb = self.item_embeddings.ID_embeddings.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

class DIFSR(nn.Module):
    def __init__(self, device, **key_words):
        super(DIFSR, self).__init__()
        
        self.cf_enhanced = key_words["cf_enhanced"]
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        self.padding_value = 0
        self.item_embeddings = Item_Embedding("DIF", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.dropout = key_words["dropout_rate"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        #self.language_dim = self.item_embeddings.language_dim
        self.hidden_dim = key_words["hidden_dim"]
        self.num_layers = key_words["num_layers"]
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.hidden_dim
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(self.dropout)
        self.emb_dropout_t = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.mh_attn = Bid_DIF_MultiHeadAttention(self.hidden_dim, self.item_embeddings.language_dim, 1, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
        #self.s_fc = nn.Linear(self.hidden_size, self.item_num)
        # self.ac_func = nn.ReLU()
        

        #self.AAP_predictor = nn.Linear(self.hidden_dim, 1, bias=True) # output 1 num, since we only use text emb, such as title.

        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_1": nn.LayerNorm(self.hidden_dim),
                "ln_1_t": nn.LayerNorm(self.language_dim),
                "ln_2": nn.LayerNorm(self.hidden_dim),
                "ln_3": nn.LayerNorm(self.hidden_dim),
                "mh_attn": Bid_DIF_MultiHeadAttention(self.hidden_dim, self.item_embeddings.language_dim, 1, self.hidden_dim, key_words["num_heads"], self.dropout),
                "ff": PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
            })
            for _ in range(self.num_layers)
        ])

    def embed_ID(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        return self.item_embeddings.ID_embeddings(x)

    def embed_text(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        return self.item_embeddings.language_embeddings(x)

    def forward(self, sequences):
        inputs_emb = self.embed_ID(sequences)
        text_emb = self.embed_text(sequences)
        #print("seq_len:", self.seq_len)
        #print("positional_embeddings size:", self.positional_embeddings.weight.shape)
        positional_embedding = self.positional_embeddings(torch.arange(inputs_emb.shape[1]).to(self.device))
        #text_emb += self.positional_embeddings(torch.arange(inputs_emb.shape[1]).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        text_seq = self.emb_dropout_t(text_emb)
        mask = torch.ne(sequences, self.padding_value).float().unsqueeze(-1).to(self.device)
        seq *= mask
        
        text_seq *= mask

        x1 = seq
        x2 = text_seq

        for layer in self.encoder_layers:
            attn_out = layer["mh_attn"](layer["ln_1"](x1), x1, positional_embedding, layer["ln_1_t"](x2),x2)
            ff_out = layer["ff"](layer["ln_2"](attn_out))
            x1 = ff_out * mask

            x1 = layer["ln_3"](x1)

        logits = x1[:, -1]  # [B, H]
        return logits
    
    def predict(self, sequences):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        state_hidden = self.forward(sequences)
        item_embs = self.item_embeddings.ID_embeddings.weight 
        scores = torch.matmul(state_hidden, item_embs.transpose(0, 1))  # (B,|I|)
        return scores
    

    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        

        batch_size = target.shape[0]
        neg_samples = torch.randint(1, self.item_num+1, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()

        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target

        target_neg = neg_samples.to(target.device)

        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        log_feats = self.forward(sequences)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)

        
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)
        
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)
        return loss

class RLMRec(Bert4Rec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("SR", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        if key_words['SR_aligement_type'] == 'con':
            self.reconstructor = nn.Sequential(
                nn.Linear(self.language_dim, (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, key_words['hidden_dim'])
            )
        elif key_words['SR_aligement_type'] == 'gen':
            self.reconstructor = nn.Sequential(
                nn.Linear(key_words['hidden_dim'], (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, self.language_dim)
            )
 
    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight
    
    def reconstruct_gen_loss(self, ):

        if self.cf_enhanced:
            rec_language_embs = self.reconstructor(self.return_item_emb()[1:] + self.item_embeddings.enhanced_cf_embeddings.weight[1:]) # self.return_item_emb()[0] is the padding embedding
        else:
            rec_language_embs = self.reconstructor(self.return_item_emb()[1:]) # self.return_item_emb()[0] is the padding embedding

        language_embs = self.item_embeddings.language_embeddings.weight
        rec_language_embs = F.normalize(rec_language_embs, p=2, dim=-1)
        language_embs = F.normalize(language_embs, p=2, dim=-1)
        return 1 - (rec_language_embs * language_embs).sum() / self.item_num
    
    def reconstruct_con_loss(self, ):
        language_embs = self.item_embeddings.language_embeddings.weight #[n+1, d_llm]
        if self.cf_enhanced:
            rec_ID_embs = self.reconstructor(language_embs) + self.item_embeddings.enhanced_cf_embeddings.weight  # self.return_item_emb()[0] is the padding embedding
        else:
            rec_ID_embs = self.reconstructor(language_embs) 
        ID_embs = self.return_item_emb()[1:]
        rec_ID_embs = F.normalize(rec_ID_embs, p=2, dim=-1)
        ID_embs = F.normalize(ID_embs, p=2, dim=-1)
        return 1 - (rec_ID_embs * ID_embs).sum() / self.item_num


class UniSRec(Bert4Rec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("AP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = MoEAdaptorLayer(
            8,
            [self.language_dim, key_words['hidden_dim']],
            0.2
        )
 
    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x) # 
        cf_embs = self.item_embeddings.ID_embeddings(x)
        return self.adapter(language_embs) + cf_embs
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        cf_embs = self.item_embeddings.ID_embeddings.weight

        return self.adapter(language_embs ) + cf_embs
     
    
class AlphaFuse(Bert4Rec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("AF", **key_words)
        #self.language_dim = self.item_embeddings.language_dim
        self.nullity = self.item_embeddings.nullity
        self.cover = key_words["cover"]
 
    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        #fuse_embs = language_embs.clone()
        ID_embs = self.item_embeddings.ID_embeddings(x)
        if self.cover:
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.nullity:] = language_embs[...,-self.nullity:] + ID_embs
        return fuse_embs
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        #fuse_embs = language_embs.clone()
        ID_embs = self.item_embeddings.ID_embeddings.weight
        if self.cover:
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.nullity:] = language_embs[...,-self.nullity:] + ID_embs
        return fuse_embs


class CF_Guard(Bert4Rec_backbone):
    def __init__(self, device, **key_words):
        #super(TFCRec, self).__init__()
        super().__init__(device, **key_words)
        self.learn_dim = key_words["learn_dim"]
        self.item_embeddings = Item_Embedding("TFC", **key_words)
        self.cf_enhanced_method = key_words["cf_enhanced_method"]
        self.language_dim = self.item_embeddings.language_dim
        if key_words["model_type"] == "TFCRec-M":
            self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, key_words['hidden_dim']),
            nn.GELU()
        )
        else:
            self.adapter = None

        self.use_eco = key_words["hot_threshold"] >= 0 and key_words["hot_threshold"] <= 0
        #self.nullity = self.item_embeddings.nullity
        self.cover = key_words["cover"]

        self.cold_items = torch.tensor(self.item_embeddings.cold_items, dtype=torch.long).to(device) #self.item_embeddings.cold_items.clone().detach().to(device)
        
        self.lamda = key_words["lamda"]
        if key_words["CF_model_type"] == "SASRec":
            self.cf_net = CFNet(device, self.item_embeddings.text_similarity, self.cold_items, **key_words).to(device)
        elif key_words["CF_model_type"] == "BERT4Rec":
            self.cf_net = CFNet_Bert(device, self.item_embeddings.text_similarity, self.cold_items, **key_words).to(device)
        elif key_words["CF_model_type"] == "GRU4Rec":
            self.cf_net = CFNet_GRU(device, self.item_embeddings.text_similarity, self.cold_items, **key_words).to(device)
        else:
            raise NotImplementedError
        #self.cf_net = CFNet(device, self.item_embeddings.text_similarity, self.cold_items, **key_words).to(device)

    def calculate_total_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        loss_1 = self.cf_net.calculate_infonce_loss( sequences, target, neg_ratio, temperature,)
        loss_2 = self.calculate_infonce_loss(sequences, target, neg_ratio, temperature,)

        return loss_1 + loss_2

    def embed_ID(self, x):
        ID_embs = self.cf_net.item_embeddings.ID_embeddings(x).detach()
        if not self.adapter:
            language_embs = self.item_embeddings.language_embeddings(x)
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.learn_dim:] = language_embs[...,-self.learn_dim:] + ID_embs 
        else:
            language_embs = self.adapter(self.item_embeddings.language_embeddings(x))
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.learn_dim:] = language_embs[...,-self.learn_dim:] + ID_embs 
        
        return fuse_embs

    def predict(self, sequences):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        state_hidden = self.forward(sequences)
        item_embs = self.return_item_emb() 
        scores = torch.matmul(state_hidden, item_embs.transpose(0, 1))  # (B,|I|)
        return scores
    
    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        
        batch_size = target.shape[0]
        neg_samples = torch.randint(1, self.item_num+1, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        #pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)

        log_feats = self.forward(sequences)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)

        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)

        return loss

    def return_item_emb(self,):
        ID_embs = self.cf_net.item_embeddings.ID_embeddings.weight.detach()
        if not self.adapter:
            language_embs = self.item_embeddings.language_embeddings.weight
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.learn_dim:] = language_embs[...,-self.learn_dim:] + ID_embs 
        else:
            language_embs = self.adapter(self.item_embeddings.language_embeddings.weight)
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.learn_dim:] = language_embs[...,-self.learn_dim:] + ID_embs 
        
        return fuse_embs
    
    def save_id_embeddings(self, path):
        #item_embs = self.return_item_emb().detach().cpu().numpy()
        id_embs = self.cf_net.item_embeddings.ID_embeddings.weight.detach().cpu().numpy()
        
        dir_path = os.path.dirname(path)
        if dir_path != "" and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(id_embs, f)
        print(f"Item embeddings saved to {path}")


