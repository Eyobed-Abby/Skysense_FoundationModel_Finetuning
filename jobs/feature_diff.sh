#!/bin/bash
#SBATCH --job-name=skysense_feature_spectru
#SBATCH --account=kuin0137

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

#SBATCH --output=jobs/logs/feature_spectra%j.out
#SBATCH --error=jobs/logs/feature_spectra%j.err

echo "======================================="
echo "   SKY SENSE FEATURE SPECTRUM ANALYSIS"
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

echo "Running feature spectral analysis..."
echo


srun python feature_diff_spectrum.py \
  --base_ckpt skysense_model_backbone_hr.pth \
  --tuned_ckpt jobs/checkpoints/sanity_model.pth \
  --train_split 0.1 \
  --batch_size 64 \
  --num_batches 10 \
  --out analysis/feature_diff_spectrum.csv


echo
echo "Finish time: $(date)"
echo "======================================="