#!/bin/bash
#SBATCH --job-name=resisc45_lora_sanity
#SBATCH --account=kuin0137

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

#SBATCH --output=jobs/logs/finetune_sanity_%j.out
#SBATCH --error=jobs/logs/finetune_sanity_%j.err

echo "======================================="
echo "   RESISC45 LoRA SANITY CHECK"
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

module purge
module load miniconda/3

cd /dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning

mkdir -p jobs/logs jobs/checkpoints

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export WANDB_API_KEY="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"


RUN_NAME="sanity_check_${SLURM_JOB_ID}"


srun python train_resisc45_lora.py \
  --ckpt skysense_model_backbone_hr.pth \
  --epochs 1 \
  --batch_size 32 \
  --train_split 0.1 \
  --save_path jobs/checkpoints/sanity_model.pth \
  --wandb_project "LoRA_resisc45_sanity"

echo
echo "Finish time: $(date)"
echo "======================================="
