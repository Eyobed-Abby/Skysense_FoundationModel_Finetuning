#!/bin/bash
#SBATCH --job-name=skysense_weight_spectrum
#SBATCH --account=kuin0137

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

#SBATCH --output=jobs/logs/weight_spectrum_%j.out
#SBATCH --error=jobs/logs/weight_spectrum_%j.err

echo "======================================="
echo "   SKY SENSE WEIGHT SPECTRUM ANALYSIS"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

module purge
module load miniconda/3

cd /dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning

mkdir -p jobs/logs analysis

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "Running weight spectral analysis..."
echo

srun python compute_update_weight.py

echo
echo "Finish time: $(date)"
echo "======================================="