#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1g
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --account=engin1  # account to charge
#SBATCH --partition=gpu # debug
##### END preamble

my_job_header
module load python3.9-anaconda
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
module load cupti/11.8.0
module list

activate


pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
pip install mmcv
pip install mmengine

pip install -r requirements.txt
pip install -v -e .

mim install "mmpose>=1.1.0"
python xxx.pyact