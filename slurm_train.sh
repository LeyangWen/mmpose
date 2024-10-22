#!/bin/bash -l
#SBATCH --job-name=RTMPose-Train
#SBATCH --output=output_slurm/train_log.txt
#SBATCH --error=output_slurm/train_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu
##### END preamble

my_job_header
#module load python3.10-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
#module load python/3.10.4
#module load pytorch/2.0.1
module list

conda activate openmmlab

# RTMPose - Train 37kpts on VEHS-7M only
python tools/train.py \
configs/wholebody_2d_keypoint/rtmpose/VEHS7M/rtmw-l_8xb320-270e_VEHS7MOnly-384x288.py \
--resume \
--wandb_name 'Test-train-slurm' \
--wandb_mode 'online' \
--arg_notes ''
