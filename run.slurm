#!/bin/bash

#SBATCH --job-name=test

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=96GB

#SBATCH --time=5-00:00:00

#SBATCH --verbose

#SBATCH --output=/scratch/%u/%x-%j.out
#SBATCH --error=/scratch/%u/%x-%j.err

module load gnu10
module load git
source /home/msu5/myenv/bin/activate
module load python/3.10
module load cuda


pip install flash-attn --no-build-isolation
pip list
cd /scratch/msu5/toyGPT_Experiments
git pull
python3 sweep.py 
