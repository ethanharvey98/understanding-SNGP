#!/bin/bash
#SBATCH --array=0-0%3
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --ntasks=16
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=gpu
#SBATCH --time=168:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    "python ../src/main_ImageNet.py --batch_size=32 --epochs=90 --experiments_directory='/cluster/tufts/hugheslab/eharve06/understanding-SNGP/experiments/ImageNet' --kappa=19.948244061859228 --lr_0=0.1 --model_name='RFF_epochs=90_kappa=19.948244061859228_lr_0=0.1_random_state=1001' --num_workers=16 --random_state=1001 --wandb --wandb_project='understanding-SNGP'"



    "python ../src/main_ImageNet.py --batch_size=32 --epochs=30 --experiments_directory='/cluster/tufts/hugheslab/eharve06/understanding-SNGP/experiments/ImageNet' --lr_0=0.1 --model_name='RFF_epochs=30_lengthscale=20.0_lr_0=0.1_outputscale=1.0_random_state=1001' --num_workers=16 --random_state=1001 --wandb --wandb_project='understanding-SNGP'"
    "python ../src/main_ImageNet.py --batch_size=32 --epochs=30 --experiments_directory='/cluster/tufts/hugheslab/eharve06/understanding-SNGP/experiments/ImageNet' --lr_0=0.01 --model_name='RFF_epochs=30_lengthscale=20.0_lr_0=0.01_outputscale=1.0_random_state=1001' --num_workers=16 --random_state=1001 --wandb --wandb_project='understanding-SNGP'"
    "python ../src/main_ImageNet.py --batch_size=32 --epochs=30 --experiments_directory='/cluster/tufts/hugheslab/eharve06/understanding-SNGP/experiments/ImageNet' --lr_0=0.001 --model_name='RFF_epochs=30_lengthscale=20.0_lr_0=0.001_outputscale=1.0_random_state=1001' --num_workers=16 --random_state=1001 --wandb --wandb_project='understanding-SNGP'"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
