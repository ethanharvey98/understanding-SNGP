#!/bin/bash
#SBATCH -n 1
#SBATCH -p ccgpu
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-2%10

source ~/.bashrc
conda activate bdl-transfer-learning

# Define an array of commands
experiments=(
    "python ../src/main_CIFAR-10.py --bb_weight_decay=1e-06 --clf_weight_decay=1e-06 --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_SNGP' --lr_0=0.01 --n=1000 --model_name='nonlearned_lr_0=0.01_n=1000_random_state=1001_weight_decay=1e-06' --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision' --prior_type='nonlearned' --random_state=1001 --save --wandb --wandb_project='retrained_CIFAR-10_SNGP'"
    "python ../src/main_CIFAR-10.py --bb_weight_decay=0.0001 --clf_weight_decay=0.0001 --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_SNGP' --lr_0=0.01 --n=1000 --model_name='nonlearned_lr_0=0.01_n=1000_random_state=2001_weight_decay=0.0001' --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision' --prior_type='nonlearned' --random_state=2001 --save --wandb --wandb_project='retrained_CIFAR-10_SNGP'"
    "python ../src/main_CIFAR-10.py --bb_weight_decay=1e-06 --clf_weight_decay=1e-06 --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_SNGP' --lr_0=0.1 --n=1000 --model_name='nonlearned_lr_0=0.1_n=1000_random_state=3001_weight_decay=1e-06' --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision' --prior_type='nonlearned' --random_state=3001 --save --wandb --wandb_project='retrained_CIFAR-10_SNGP'"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate