#!/bin/sh
#SBATCH --job-name=diffae_test          # Name des Jobs auf dem Cluster
#SBATCH --output=output_diffae.log      # Ausgabedatei
#SBATCH --error=errors_diffae.log       # Fehlerdatei
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


srun --time=1:00:00 --gres=gpu:volta:2 python run_mri.py
