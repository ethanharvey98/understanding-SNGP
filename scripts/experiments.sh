#!/bin/bash
#SBATCH -n 16
#SBATCH -p gpu
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-0%1

source ~/.bashrc
conda activate bdl-transfer-learning

# Define an array of commands
experiments=(
    "python ../src/main.py --epochs=30 --experiments_directory='/cluster/tufts/hugheslab/eharve06/understanding-SNGP/experiments/ImageNet'  --lr_0=0.001 --model_name='PyTorch_SNGP_random_state=1001_spec_norm_bound=6.0_tau=1e-6' --random_state=1001 --tau=1e-6 --wandb --wandb_project='understanding-SNGP'"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate