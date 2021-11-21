#!/bin/bash --login
#SBATCH --job-name=kaggle-dogv0
#SBATCH --error=%j%xe.txt
#SBATCH --output=%j%xo.txt
#SBATCH --partition=gpu
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:2
#SBATCH --account=scw1875
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zhao-ziang@qq.com 

clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_3_exclusive"

module purge
module load pytorch
module load CUDA
module list

python kaggle-dog.py
