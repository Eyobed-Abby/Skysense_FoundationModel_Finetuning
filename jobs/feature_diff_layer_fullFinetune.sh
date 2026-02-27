#!/bin/bash
#SBATCH --job-name=one_epoch_full_ft_diff
#SBATCH --account=kuin0137

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

#SBATCH --output=/dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning/jobs/logs/one_epoch_full_ft_%j.out
#SBATCH --error=/dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning/jobs/logs/one_epoch_full_ft_%j.err

echo "======================================="
echo " SKY SENSE ONE-EPOCH FULL FINETUNE FEATURE-DIFF ANALYSIS"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

module purge
module load miniconda/3

cd /dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning

mkdir -p jobs/logs
mkdir -p analysis

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "Running one-epoch full finetune + feature-diff intrinsic-dimension analysis..."
echo

srun python feature_diff_full_finetune_analysis.py \
  --base_ckpt skysense_model_backbone_hr.pth \
  --train_split 0.1 \
  --batch_size 64 \
  --train_epochs 1 \
  --max_train_batches -1 \
  --analysis_batches 20 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --seed 42 \
  --out analysis/one_epoch_full_ft_feature_diff.csv

echo
echo "Finish time: $(date)"
echo "======================================="