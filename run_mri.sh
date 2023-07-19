#!/usr/bin/zsh

#SBATCH --job-name=diffae_autoenc5
#SBATCH --output=output5.log
#SBATCH --error=errors5.log
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=04:00:00   

module load CUDA

# Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/anaconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
# Now you can activate your configured conda environments
conda activate diffautoenc

echo; export; echo; nvidia-smi; echo


export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python test_manipulate.py
