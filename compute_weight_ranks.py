import torch
import torch.nn as nn
from torch.linalg import matrix_rank


def compute_ranks(model: nn.Module):
    print("\nMatrix Rank Report for Linear Layers\n")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            rank = matrix_rank(weight).item()
            print(f"{name:60} | shape: {tuple(weight.shape):>15} | rank: {rank}")


if __name__ == "__main__":
    import argparse
    from skysense_lora_classifier_qkv import build_lora_classifier_qkv

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to SkySense HR checkpoint', default="skysense_model_backbone_hr.pth")
    args = parser.parse_args()

    model = build_lora_classifier_qkv(args.ckpt)
    compute_ranks(model)
