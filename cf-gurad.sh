# __________________________SASRec backbone_________________________
split_mode=time
data=movies
algs=("CF-Guard")
learn_dims=(128)
alpha=0.3
lambda_cold=0.2
temps=1.0
Model_type=("SASRec")
for alg in "${algs[@]}"; 
    do
        for model_type in "${Model_type[@]}"; 
        do
            for learn_dim in "${learn_dims[@]}"; 
            do
            python -u train.py    --data $data --lambda_cold $lambda_cold --CF_model_type $model_type  --learn_dim $learn_dim --alpha $alpha  --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal --SR_aligement_type con --hidden_dim 128 --null_dim 64 --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 > log/${split_mode}/$data/SASRec_CFR${model_type}_CFRlr${learn_dim}_${alg}_rs22_dim128_infoNCE64_lr3_normal_init_withdrop0.1.log
            echo "done $alg"
            done
        done
done


split_mode=time
data=games
algs=("CF-Guard")
learn_dims=(32)
alpha=5.0
lambda_cold=0.2
temps=1.0
Model_type=("SASRec")
for alg in "${algs[@]}"; 
    do
        for model_type in "${Model_type[@]}"; 
        do
            for learn_dim in "${learn_dims[@]}"; 
            do
            python -u train.py    --data $data --lambda_cold $lambda_cold --CF_model_type $model_type  --learn_dim $learn_dim --alpha $alpha  --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal --SR_aligement_type con --hidden_dim 128 --null_dim 64 --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 > log/${split_mode}/$data/SASRec_CFR${model_type}_CFRlr${learn_dim}_${alg}_rs22_dim128_infoNCE64_lr3_normal_init_withdrop0.1.log
            echo "done $alg"
            done
        done
done



split_mode=time
data=baby
algs=("CF-Guard")
learn_dims=(16)
alpha=0.5
lambda_cold=0.5
temps=1.0
Model_type=("SASRec")
for alg in "${algs[@]}"; 
    do
        for model_type in "${Model_type[@]}"; 
        do
            for learn_dim in "${learn_dims[@]}"; 
            do
            python -u train.py    --data $data --lambda_cold $lambda_cold --CF_model_type $model_type --learn_dim $learn_dim --alpha $alpha  --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal --SR_aligement_type con --hidden_dim 128 --null_dim 64 --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 > log/${split_mode}/$data/SASRec_CFR${model_type}_CFRlr${learn_dim}_${alg}_rs22_dim128_infoNCE64_lr3_normal_init_withdrop0.1.log
            echo "done $alg"
            done
        done
done


# __________________________Bert4Rec backbone_________________________

split_mode=time
data=movies
algs=("CF-Guard")
learn_dims=(128)
alpha=0.5
lambda_cold=0.2
temps=1.0
Model_type=("Bert4Rec")
for alg in "${algs[@]}"; 
    do
        for model_type in "${Model_type[@]}"; 
        do
            for learn_dim in "${learn_dims[@]}"; 
            do
            python -u train_bert.py    --data $data --lambda_cold $lambda_cold --CF_model_type $model_type --learn_dim $learn_dim --alpha $alpha  --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal --SR_aligement_type con --hidden_dim 128 --null_dim 64 --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 > log/${split_mode}/$data/Bert4Rec_CFR${model_type}_CFRlr${learn_dim}_${alg}_rs22_dim128_infoNCE64_lr3_normal_init_withdrop0.1.log
            echo "done $alg"
            done
        done
done


split_mode=time
data=games
algs=("CF-Guard")
learn_dims=(64)
alpha=5.0
lambda_cold=0.2
temps=1.0
Model_type=("Bert4Rec")
for alg in "${algs[@]}"; 
    do
        for model_type in "${Model_type[@]}"; 
        do
            for learn_dim in "${learn_dims[@]}"; 
            do
            python -u train_bert.py    --data $data --lambda_cold $lambda_cold --CF_model_type $model_type --learn_dim $learn_dim --alpha $alpha  --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal --SR_aligement_type con --hidden_dim 128 --null_dim 64 --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 > log/${split_mode}/$data/Bert4Rec_CFR${model_type}_CFRlr${learn_dim}_${alg}_rs22_dim128_infoNCE64_lr3_normal_init_withdrop0.1.log
            echo "done $alg"
            done
        done
done



split_mode=time
data=baby
algs=("CF-Guard")
learn_dims=(32)
alpha=0.5
lambda_cold=0.2
temps=1.0
Model_type=("Bert4Rec")
for alg in "${algs[@]}"; 
    do
        for model_type in "${Model_type[@]}"; 
        do
            for learn_dim in "${learn_dims[@]}"; 
            do
            python -u train_bert.py    --data $data --lambda_cold $lambda_cold --CF_model_type $model_type --learn_dim $learn_dim --alpha $alpha  --random_seed 22  --model_type $alg --cuda 2 --split_mode $split_mode --language_model_type qwen3 --ID_embs_init_type normal --SR_aligement_type con --hidden_dim 128 --null_dim 64 --lr 0.001 --loss_type infoNCE --dropout_rate 0.1 --neg_ratio 64 > log/${split_mode}/$data/Bert4Rec_CFR${model_type}_CFRlr${learn_dim}_${alg}_rs22_dim128_infoNCE64_lr3_normal_init_withdrop0.1.log
            echo "done $alg"
            done
        done
done