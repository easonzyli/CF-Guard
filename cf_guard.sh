#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

split_mode="time"
cuda_id="${CUDA_ID:-2}"
python_bin="${PYTHON_BIN:-python}"

seeds=(22 2023 2024 2025 2026)
datasets=("movies" "games" "baby")

for seed in "${seeds[@]}"; do
    for backbone in "SASRec" "Bert4Rec"; do
        for data in "${datasets[@]}"; do
            if [ "$backbone" = "SASRec" ]; then
                train_script="train.py"
                case "$data" in
                    movies)
                        learn_dim=128
                        alpha=0.3
                        lambda_cold=0.2
                        ;;
                    games)
                        learn_dim=32
                        alpha=5.0
                        lambda_cold=0.2
                        ;;
                    baby)
                        learn_dim=16
                        alpha=0.5
                        lambda_cold=0.5
                        ;;
                esac
            else
                train_script="train_bert.py"
                case "$data" in
                    movies)
                        learn_dim=128
                        alpha=0.5
                        lambda_cold=0.2
                        ;;
                    games)
                        learn_dim=64
                        alpha=5.0
                        lambda_cold=0.2
                        ;;
                    baby)
                        learn_dim=32
                        alpha=0.5
                        lambda_cold=0.2
                        ;;
                esac
            fi

            log_dir="log/${split_mode}/${data}"
            mkdir -p "$log_dir"
            log_file="${log_dir}/${backbone}_CFR${backbone}_CFRlr${learn_dim}_CF-Guard_rs${seed}_dim128_infoNCE64_lr3_normal_init_withdrop0.1.log"

            echo "Running dataset=${data}, backbone=${backbone}, seed=${seed}, cuda=${cuda_id}"
            "$python_bin" -u "$train_script" \
                --data "$data" \
                --lambda_cold "$lambda_cold" \
                --CF_model_type "$backbone" \
                --learn_dim "$learn_dim" \
                --alpha "$alpha" \
                --random_seed "$seed" \
                --model_type "CF-Guard" \
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

            echo "done CF-Guard, dataset=${data}, backbone=${backbone}, seed=${seed}"
        done
    done
done
