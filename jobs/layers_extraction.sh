#!/bin/bash
#SBATCH --job-name=layer_analysis
#SBATCH --account=kuin0137

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

#SBATCH --output=/dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning/jobs/logs/layer_analysis_%j.out
#SBATCH --error=/dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning/jobs/logs/layer_analysis_%j.err

echo "======================================="
echo " SKY SENSE LAYER-WISE FEATURE ANALYSIS "
echo "======================================="
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo

module purge
module load miniconda/3

cd /dpc/kuin0137/LoRA_Experiment/classification/Updated_Vanilla/Skysense_FoundationModel_Finetuning

# Ensure directories exist
mkdir -p jobs/logs
mkdir -p analysis

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "Running full feature-diff intrinsic-dimension analysis..."
echo

srun python export_skysense_layers_to_csv.py \
  --base_ckpt skysense_model_backbone_hr.pth \
  --out_modules analysis/backbone_modules.csv \
  --out_params analysis/backbone_params.csv

echo
echo "Finish time: $(date)"
echo "======================================="