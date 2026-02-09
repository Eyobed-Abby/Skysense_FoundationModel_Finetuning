# === file: analysis/compute_weight_ranks.py ===

import torch
import torch.nn as nn
from torch.linalg import matrix_rank
import csv
import os

def compute_ranks(model: nn.Module, save_path: str):
    print("\nMatrix Rank Report for Linear Layers\n")
    rows = [("layer", "group", "shape", "rank")]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            rank = matrix_rank(weight).item()
            shape = tuple(weight.shape)

            # Determine group label
            if ".qkv" in name:
                group = "qkv"
            elif ".ffn" in name:
                group = "ffn"
            elif ".proj" in name:
                group = "proj"
            else:
                group = "other"

            print(f"{name:60} | shape: {str(shape):>15} | rank: {rank}")
            rows.append((name, group, str(shape), rank))

    # Save CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nSaved rank report to {save_path}")


if __name__ == "__main__":
    import argparse
    from skysense_lora_classifier_qkv import build_lora_classifier_qkv

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to SkySense HR checkpoint')
    parser.add_argument('--out', type=str, default="analysis/weight_ranks.csv", help='Output CSV path')
    args = parser.parse_args()

    model = build_lora_classifier_qkv(args.ckpt)
    compute_ranks(model, args.out)