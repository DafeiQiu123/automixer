#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding -c 4 -N 1
#SBATCH -J 2470

#SBATCH -o tmp.log

bash init.sh
bash train.sh
