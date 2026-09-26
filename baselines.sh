#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

split_mode="time"
cuda_id="${CUDA_ID:-2}"
python_bin="${PYTHON_BIN:-python}"

seeds=(22 2023 2024 2025 2026)
datasets=("movies" "games" "baby")
algs=("SASRec" "LLMEmb" "RLMRec" "MoRec" "UniSRec" "DIF-SR" "TedRec" "AlphaFuse" "LLMInit")

for seed in "${seeds[@]}"; do
    for data in "${datasets[@]}"; do
        log_dir="log/${split_mode}/${data}"
        mkdir -p "$log_dir"

        for alg in "${algs[@]}"; do
            log_file="${log_dir}/SASRec_${alg}_rs${seed}_dim128_infoNCE64_lr3_normal_init_withdrop0.1.log"

            echo "Running dataset=${data}, model=${alg}, seed=${seed}, cuda=${cuda_id}"
            "$python_bin" -u train.py \
                --data "$data" \
                --num_layers2 2 \
                --random_seed "$seed" \
                --model_type "$alg" \
                --cuda "$cuda_id" \
                --split_mode "$split_mode" \
                --language_model_type qwen3 \
                --ID_embs_init_type normal \
                --SR_aligement_type con \
                --hidden_dim 128 \
                --null_dim 64 \
                --lr 0.001 \
                --loss_type infoNCE \
                --dropout_rate 0.1 \
                --neg_ratio 64 \
                > "$log_file" 2>&1

            echo "done ${alg}, dataset=${data}, seed=${seed}"
        done
    done
done
