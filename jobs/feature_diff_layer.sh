#!/bin/bash
#SBATCH --job-name=skysense_feature_diff
#SBATCH --account=kuin0137

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

#SBATCH --output=/dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning/jobs/logs/feature_diff_%j.out
#SBATCH --error=/dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning/jobs/logs/feature_diff_%j.err

echo "======================================="
echo " SKY SENSE LAYER-WISE FEATURE DIFF SPECTRUM"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

module purge
module load miniconda/3

cd /dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning

# Ensure directories exist BEFORE execution
mkdir -p jobs/logs
mkdir -p analysis

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "Running feature difference layer-wise spectral analysis..."
echo

srun python feature_diff_block_analysis.py \
  --base_ckpt skysense_model_backbone_hr.pth \
  --tuned_ckpt jobs/checkpoints/sanity_model.pth \
  --train_split 0.1 \
  --batch_size 64 \
  --num_batches 10 \
  --out analysis/feature_diff_block_analysis.csv

echo
echo "Finish time: $(date)"
echo "======================================="