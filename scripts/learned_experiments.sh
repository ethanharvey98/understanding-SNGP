#!/bin/bash
#SBATCH -n 1
#SBATCH -p hugheslab
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-2
source ~/.bashrc
conda activate bdl-transfer-learning

# Define an array of commands
experiments=(
    "python ../src/main_CIFAR-10.py --bb_weight_decay=0.01 --clf_weight_decay=0.01 --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_multimodal' --lr_0=0.1 --m=10 --n=50000 --model_name='adapted_lr_0=0.1_m=10_n=50000_random_state=1001_weight_decay=0.01' --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision' --prior_type='adapted' --random_state=1001 --save --wandb --wandb_project='retrained_CIFAR-10_multimodal'"
    "python ../src/main_CIFAR-10.py --bb_weight_decay=0.01 --clf_weight_decay=0.01 --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_multimodal' --lr_0=0.1 --m=10 --n=50000 --model_name='adapted_lr_0=0.1_m=10_n=50000_random_state=2001_weight_decay=0.01' --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision' --prior_type='adapted' --random_state=2001 --save --wandb --wandb_project='retrained_CIFAR-10_multimodal'"
    "python ../src/main_CIFAR-10.py --bb_weight_decay=0.01 --clf_weight_decay=0.01 --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_multimodal' --lr_0=0.1 --m=10 --n=50000 --model_name='adapted_lr_0=0.1_m=10_n=50000_random_state=3001_weight_decay=0.01' --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision' --prior_type='adapted' --random_state=3001 --save --wandb --wandb_project='retrained_CIFAR-10_multimodal'"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate