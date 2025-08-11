#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --mem=6400MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/data3/yoorheekim/NLP_Study/Week_3/outputs/sanity_check_1d.log
#SBATCH --error=/data3/yoorheekim/NLP_Study/Week_3/outputs/sanity_check_1d.err

source /home/yoorheekim/.bashrc
source /data3/yoorheekim/miniconda3/etc/profile.d/conda.sh

conda activate a3q3

srun python sanity_check.py 1d