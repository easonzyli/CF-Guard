#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

split_mode="time"
cuda_id="${CUDA_ID:-2}"
python_bin="${PYTHON_BIN:-python}"
seed=22

datasets=("movies" "baby")
modes=("language" "random" "shuffle" "zero")

for data in "${datasets[@]}"; do
    log_dir="log/${split_mode}/${data}/controlled"
    mkdir -p "$log_dir"

    for mode in "${modes[@]}"; do
        log_file="${log_dir}/SASRec_AlphaFuse_${mode}_rs${seed}_dim128_infoNCE64_lr0.001_drop0.1.log"

        echo "Running dataset=${data}, seed=${seed}, mode=${mode}, cuda=${cuda_id}"
        "$python_bin" -u train_controlled.py \
            --data "$data" \
            --split_mode "$split_mode" \
            --model_type "AlphaFuse" \
            --controlled_mode "$mode" \
            --random_seed "$seed" \
            --cuda "$cuda_id" \
            --language_model_type "qwen3" \
            --ID_embs_init_type "normal" \
            --SR_aligement_type "con" \
            --hidden_dim 128 \
            --null_dim 64 \
            --lr 0.001 \
            --loss_type "infoNCE" \
            --dropout_rate 0.1 \
            --neg_ratio 64 \
            --temperature 0.07 \
            > "$log_file" 2>&1

        status=$?
        if [ "$status" -ne 0 ]; then
            echo "FAILED dataset=${data}, seed=${seed}, mode=${mode}; see ${log_file}"
        else
            echo "Done dataset=${data}, seed=${seed}, mode=${mode}"
        fi
    done
done
