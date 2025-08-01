#!/bin/bash
#SBATCH --job-name=skipgram_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-24:00:00
#SBATCH --mem=6400MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/data3/yoorheekim/NLP_Study/Week_2/skipgram/jupyter_%j.log
#SBATCH --error=/data3/yoorheekim/NLP_Study/Week_2/skipgram/jupyter_%j.err

source /home/yoorheekim/.bashrc
source /data3/yoorheekim/miniconda3/etc/profile.d/conda.sh

conda activate torch_env

srun jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888