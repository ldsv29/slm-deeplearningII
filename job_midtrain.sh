#!/bin/bash
#SBATCH --job-name=slm_midtrain
#SBATCH --output=slm_midtrain.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=seu-email@edu.pucrs.br
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH -t 4:00:00

echo "Job iniciado em $(date)"
echo "Rodando no nó: $(hostname)"

source /scratch/ap212/slm/.venv/bin/activate

cd /home/ap212/slm-deeplearningII

torchrun --nproc_per_node=2 train.py

echo "Job finalizado em $(date)"