#!/usr/bin/env bash
set -euo pipefail

# 参数网格
model_d_models=(32 128 512)            # --model_d_model
model_layers_list=(1)                 # --model_layers
positional_encodings=(none naive bpm)   # --use_pos_encoding

for d in "${model_d_models[@]}"; do
  for layers in "${model_layers_list[@]}"; do
    for pe in "${positional_encodings[@]}"; do
      ckpt_dir="exp/checkpoints_d${d}_L${layers}_pe_${pe}"
      plot_path="${ckpt_dir}/training_curve_d${d}_L${layers}_pe_${pe}.png"
      .venv/bin/python models/data_pipeline.py \
        --input_dir data/wav_dir_trimmed_split \
        --pairing_mode random \
        --num_pairs_train 5000 \
        --num_pairs_valid 100 \
        --use_pos_encoding "${pe}" \
        --model_d_model "${d}" \
        --model_layers "${layers}" \
        --epochs 3 \
        --plot_path "${plot_path}" \
        --batch_size 8 \
        --valid_ratio 0.1 \
        --ckpt_dir "${ckpt_dir}"
      echo "${d}_${layers}_${pe} [done]" >> log
    done
  done
done
