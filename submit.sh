#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH -p gpu --gres=gpu:2 --gres-flags=enforce-binding -c 8 -N 1
#SBATCH -J 2470

#SBATCH -o tmp.log

nvidia-smi
bash init.sh
bash train.sh
