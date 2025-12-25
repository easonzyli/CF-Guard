# __________________________SASRec backbone_________________________
split_mode=time
data=movies
algs=("SASRec" "LLMEmb" "RLMRec" "MoRec" "UniSRec" "DIF-SR" "TedRec" "AlphaFuse"  "LLMInit")
for alg in "${algs[@]}"; 
    do
        python -u train.py    --data $data  --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal  --hidden_dim 128  --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 
        echo "done $alg"
    done


split_mode=time
data=games
algs=("SASRec" "LLMEmb" "RLMRec" "MoRec" "UniSRec" "DIF-SR" "TedRec" "AlphaFuse"  "LLMInit")
for alg in "${algs[@]}"; 
    do
        python -u train.py    --data $data --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal  --hidden_dim 128  --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 
        echo "done $alg"
    done



split_mode=time
data=baby
algs=("SASRec" "LLMEmb" "RLMRec" "MoRec" "UniSRec" "DIF-SR" "TedRec" "AlphaFuse"  "LLMInit")
for alg in "${algs[@]}"; 
    do
        python -u train.py    --data $data   --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal  --hidden_dim 128  --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 
        echo "done $alg"
    done

# __________________________BERT4Rec backbone_________________________
split_mode=time
data=movies
algs=("BERT4Rec" "LLMEmb" "RLMRec" "MoRec" "UniSRec" "DIF-SR" "TedRec" "AlphaFuse"  "LLMInit")
for alg in "${algs[@]}"; 
    do
        python -u train_bert.py    --data $data  --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal  --hidden_dim 128  --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 
        echo "done $alg"
    done


split_mode=time
data=games
algs=("BERT4Rec" "LLMEmb" "RLMRec" "MoRec" "UniSRec" "DIF-SR" "TedRec" "AlphaFuse"  "LLMInit")
for alg in "${algs[@]}"; 
    do
        python -u train_bert.py    --data $data --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal  --hidden_dim 128  --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 
        echo "done $alg"
    done



split_mode=time
data=baby
algs=("BERT4Rec" "LLMEmb" "RLMRec" "MoRec" "UniSRec" "DIF-SR" "TedRec" "AlphaFuse"  "LLMInit")
for alg in "${algs[@]}"; 
    do
        python -u train_bert.py    --data $data   --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal  --hidden_dim 128  --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 
        echo "done $alg"
    done