#!/bin/bash

# TOPK_VALUES=(128 256 512 1024 2048 4096)

# for k in "${TOPK_VALUES[@]}"; do
#     echo "Running top_k=${k}"
#     python topk_sae.py \
#         --input_pkl train_data/mlp_act_world_10k_l16.pkl \
#         --top_k "${k}" \
#         > "training_log/log_topksae_${k}.txt" 2>&1
# done

# echo "All runs completed."



N_VALUES=(1 2 4 8 16)

k="$1"

mkdir "training_log/top${k}_dimN"
mkdir "models/top${k}_dimN"

echo "Create dirs for top${k}_dimN"

echo "Training SAE for k=${k}"

for n in "${N_VALUES[@]}"; do
    echo "Running dim N=${n}"
    python topk_sae.py \
        --input_pkl train_data/world10k/mlp_act_world_10k_l16.pkl \
        --top_k "$k" \
        --output_dir models/top${k}_dimN/ \
        --hidden_dim "${n}" \
        > "training_log/top${k}_dimN/log_top${k}_N${n}.txt" 2>&1
done

echo "All runs completed."
