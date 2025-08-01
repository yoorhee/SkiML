#!/bin/bash
#SBATCH --job-name=skipgram_b_1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --mem=6400MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/data3/yoorheekim/NLP_Study/Week_2/real_skipgram/results/ass_1_b.log
#SBATCH --error=/data3/yoorheekim/NLP_Study/Week_2/real_skipgram/results/ass_1_b.err

source /home/yoorheekim/.bashrc
source /data3/yoorheekim/miniconda3/etc/profile.d/conda.sh

conda activate a2

srun python word2vec.py naiveSoftmaxLossAndGradient