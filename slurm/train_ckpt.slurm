#!/bin/bash
#SBATCH --job-name=tex_diff_ipt
#SBATCH --partition=ckpt
#SBATCH --account=realitylab-ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gpus=a40:4
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifan1@uw.edu

# singularity container path
SIF_PATH=/gscratch/realitylab/yifan1/apptainer/images/ldm.sif

# cmd to run in singularity
CMD="cd /gscratch/realitylab/yifan1/latent-diffusion;python main.py --base configs/latent-diffusion/tex-ldm-vq-f8-ipt.yaml -r logs/2022-10-07T00-47-19_tex-ldm-vq-f8-ipt/ -t --gpus 0,1,2,3,"

# load singularity
module load apptainer

# execute training using singularity
apptainer exec --nv --bind /gscratch/realitylab/yifan1 ${SIF_PATH} bash -c "${CMD}"

