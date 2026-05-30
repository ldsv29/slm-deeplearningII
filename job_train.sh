#!/bin/bash
#SBATCH --job-name=slm_pretrain
#SBATCH --output=slm_pretrain.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=seu-email@edu.pucrs.br
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH -t 08:00:00

echo "Job iniciado em $(date)"
echo "Rodando no nó: $(hostname)"

source ~/.venv/bin/activate

cd ~/seu-repo

torchrun --nproc_per_node=2 train.py

echo "Job finalizado em $(date)"