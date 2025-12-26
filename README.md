This is the official implementation of the paper "Lost in Language: Your Language Embeddings Undermine ID Embeddings Learning in Sequential Recommendation" 

# Overview

The main implementation of our proposed CF-Guard can be found in the file `models/backbone_SASRec.py` and `models/backbone_BERT4Rec.py`. 


# Quick Start
To reproduce the results reported in the paper, please follow these steps: 
## ​ 1. Download Datasets

https://drive.google.com/file/d/1OKhI0kByn0ZuO0x_oE_ld3_L4raKbG1R/view?usp=sharing

Please download our datasets and unzip.  The directory structure should be as follows:

```text
data
├── time
    ├── baby
    │   ├── data_statis.df
    │   ├── data_test.txt
    │   ├── data_train.txt
    │   ├── data_valid.txt
    │   ├── id2title.json
    │   ├── item_freq.json
    │   ├── item2id.json
    │   └── qwen3_embeddings.pkl
    ├── games
    │   └── (same structure as `baby`)
    └── movies
        └── (same structure as `baby`)

```

## 2. Running CF-Guard

### SASRec backbone 

```sh
# games
python train.py --data games --cuda 2  --random_seed 22 --model_type CF-Guard  --hidden_dim 128 --CF_model_type SASRec   --lambda_cold 0.2  --learn_dim 32 --alpha 5.0  

# movies
python train.py --data movies --cuda 2  --random_seed 22 --model_type CF-Guard  --hidden_dim 128 --CF_model_type SASRec   --lambda_cold 0.2  --learn_dim 128 --alpha 0.3  

# baby
python train.py --data baby --cuda 2  --random_seed 22 --model_type CF-Guard  --hidden_dim 128 --CF_model_type SASRec   --lambda_cold 0.2  --learn_dim 16 --alpha 0.5 
```
### Bert4Rec backbone
```sh
# games
python train_bert.py --data games --cuda 2  --random_seed 22 --model_type CF-Guard  --hidden_dim 128 --CF_model_type BERT4Rec  --lambda_cold 0.2  --learn_dim 64 --alpha 5.0  

# movies
python train_bert.py --data movies --cuda 2  --random_seed 22 --model_type CF-Guard  --hidden_dim 128 --CF_model_type BERT4Rec   --lambda_cold 0.2  --learn_dim 128 --alpha 5.0  

# baby
python train_bert.py --data baby --cuda 2  --random_seed 22 --model_type CF-Guard  --hidden_dim 128 --CF_model_type BERT4Rec   --lambda_cold 1.0  --learn_dim 32 --alpha 0.5  
```

## 3. Running Baselines

### SASRec backbone & Bert4Rec backbone

```sh
bash baselines.sh
```


## Hyperparameters for Timestamp Settings

### SASRec backbone

| Dataset    | $\lambda_{cold}$  | $d_c$    | $\alpha$  | 
| ---------- | ------------- | ------------ | ------ |
| **Games** | 0.2           | 32           | 5.0    |      
| **Movies**   | 0.2        | 128           | 0.3    |     
| **Baby** | 0.5           | 16          | 0.5    |   

### Bert4Rec backbone

| Dataset    | $\lambda_{cold}$ | $d_c$    | $\alpha$  | 
| ---------- | ------------- | ------------ | ------ | 
| **Games** | 0.2           | 64           | 5.0    |   
| **Movies**   | 0.2        | 128           | 5.0    |   
| **Baby** | 1.0           | 32           | 0.5    |    


## Environments

We conducted the experiments based on the following environments:
* CUDA Version: 13.0
* OS: Ubuntu 18.04.5 LTS (Bionic Beaver)
* GPU: The NVIDIA 3090 GPU
* CPU: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz

Our code is implemented based on [RecBole](https://github.com/RUCAIBox/RecBole) and [AlphaFuse](https://github.com/Hugo-Chinn/AlphaFuse).
