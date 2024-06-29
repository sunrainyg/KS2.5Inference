#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=96:00:00
#SBATCH --partition=cbmm

# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

python inference.py --do_ocr

