#!/bin/bash
#SBATCH --job-name=testrun
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --mem=6400MB
#SBATCH --cpus-per-task=1
#SBATCH --output=/data3/yoorheekim/test_folder/result_test.log
#SBATCH --error=/data3/yoorheekim/test_folder/error_test.log

source /home/yoorheekim/.bashrc
source /data3/yoorheekim/miniconda3/etc/profile.d/conda.sh
conda activate base

srun python test.py