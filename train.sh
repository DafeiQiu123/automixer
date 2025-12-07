#!/usr/bin/env bash
set -euo pipefail

# 参数网格
model_d_models=(32 128 512)            # --model_d_model
model_layers_list=(1)                  # --model_layers
positional_encodings=(none naive bpm)  # --use_pos_encoding

# 生成 9 个任务
tasks=()
for d in "${model_d_models[@]}"; do
  for layers in "${model_layers_list[@]}"; do
    for pe in "${positional_encodings[@]}"; do
      tasks+=("${d}|${layers}|${pe}")
    done
  done
done

# 按 4 个和 5 个拆分给两张 GPU
tasks_gpu0=("${tasks[@]:0:4}")
tasks_gpu1=("${tasks[@]:4}")

pids=()

# GPU0 顺序执行自己的 4 个任务；与 GPU1 并行
(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=0
  for t in "${tasks_gpu0[@]}"; do
    IFS='|' read -r d layers pe <<<"$t"
    ckpt_dir="exp/checkpoints_d${d}_L${layers}_pe_${pe}"
    plot_path="${ckpt_dir}/training_curve_d${d}_L${layers}_pe_${pe}.png"
    mkdir -p "${ckpt_dir}"
    .venv/bin/python models/data_pipeline.py \
      --input_dir data/wav_dir_trimmed_split \
      --pairing_mode random \
      --num_pairs_train 5000 \
      --num_pairs_valid 100 \
      --use_pos_encoding "${pe}" \
      --model_d_model "${d}" \
      --iter_group 20 \
      --model_layers "${layers}" \
      --epochs 3 \
      --plot_path "${plot_path}" \
      --batch_size 8 \
      --valid_ratio 0.1 \
      --ckpt_dir "${ckpt_dir}"
    echo "${d}_${layers}_${pe} [done]" >> log
  done
) &
pids+=($!)

# GPU1 顺序执行自己的 5 个任务；与 GPU0 并行
(
  set -euo pipefail
  export CUDA_VISIBLE_DEVICES=1
  for t in "${tasks_gpu1[@]}"; do
    IFS='|' read -r d layers pe <<<"$t"
    ckpt_dir="exp/checkpoints_d${d}_L${layers}_pe_${pe}"
    plot_path="${ckpt_dir}/training_curve_d${d}_L${layers}_pe_${pe}.png"
    mkdir -p "${ckpt_dir}"
    .venv/bin/python models/data_pipeline.py \
      --input_dir data/wav_dir_trimmed_split \
      --pairing_mode random \
      --num_pairs_train 5000 \
      --num_pairs_valid 100 \
      --use_pos_encoding "${pe}" \
      --model_d_model "${d}" \
      --iter_group 20 \
      --model_layers "${layers}" \
      --epochs 3 \
      --plot_path "${plot_path}" \
      --batch_size 8 \
      --valid_ratio 0.1 \
      --ckpt_dir "${ckpt_dir}"
    echo "${d}_${layers}_${pe} [done]" >> log
  done
) &
pids+=($!)

# 等待两张卡任务结束；若有失败则退出
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "并行任务失败: PID=${pid}" >&2
    exit 1
  fi
done
