#!/bin/bash -l
#SBATCH --job-name=RTMPose-Train
#SBATCH --output=output_slurm/train_log.txt
#SBATCH --error=output_slurm/train_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu

##### END preamble

#Sxxxxx --gpu-bind=single:1

my_job_header
#module load python3.10-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
#module load python/3.10.4
#module load pytorch/2.0.1
module list

conda activate openmmlab

nvidia-smi

## RTMPose - Train 37kpts on VEHS-7M only
#srun python tools/train.py \
#configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7MOnly-384x288.py \
#--wandb_name 'Train-slurm-VEHS7MOnly' \
#--wandb_mode 'online' \
#--arg_notes 'VEHS-7M only - resume' \
#--resume \

#CONFIG="configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7MOnly-384x288.py"
CONFIG="configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7Mplus-384x288.py"
GPUS=$SLURM_GPUS_ON_NODE
NNODES=$SLURM_NNODES
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --wandb_name 'Train-slurm-VEHS7Plus' \
    --wandb_mode 'online' \
    --arg_notes 'VEHS-7M plus - 3GPU - speed test' \
    --launcher pytorch
#        --resume \

