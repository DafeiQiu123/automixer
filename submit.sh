#!/bin/bash
set -euo pipefail

# 获取当前脚本所在目录，确保相对路径稳健
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 使用 1 GPU、4 核、24 小时提交 train.sh；其他配置以命令行为准覆盖脚本内的 #SBATCH
JOB_ID=$(sbatch --parsable \
  --time=24:00:00 \
  -p gpu \
  --gres=gpu:1 \
  -c 4 \
  "${SCRIPT_DIR}/train.sh")

echo "Submitted job ${JOB_ID}"