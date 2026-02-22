import os
import re
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from resisc45_loader import get_resisc45_dataloaders
from skysense_lora_classifier_qkv import build_lora_classifier_qkv


# ---------------------------------------------------------
# 1. Identify Swin Block Outputs
# ---------------------------------------------------------

def is_swin_block(name):
    # match: backbone.stages.X.blocks.Y
    pattern = r"backbone\.stages\.\d+\.blocks\.\d+$"
    return re.search(pattern, name) is not None


def register_block_hooks(model):
    activations = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            out = output
            if isinstance(out, tuple):
                out = out[0]

            # Expect [B, C, H, W]
            if out.ndim == 4:
                out = nn.functional.adaptive_avg_pool2d(out, 1)
                out = out.squeeze(-1).squeeze(-1)  # [B, C]
            elif out.ndim == 3:
                # [B, L, C] → mean over tokens
                out = out.mean(dim=1)
            else:
                return

            out = out.detach().cpu()
            activations.setdefault(name, []).append(out)

        return hook

    for name, module in model.named_modules():
        if is_swin_block(name):
            handles.append(module.register_forward_hook(make_hook(name)))

    return activations, handles


# ---------------------------------------------------------
# 2. Spectrum Computation
# ---------------------------------------------------------

def compute_spectrum(M):
    M = M.float()
    U, S, V = torch.linalg.svd(M, full_matrices=False)
    eigvals = S**2
    total = eigvals.sum()
    ratios = eigvals / total
    cumulative = torch.cumsum(ratios, dim=0)

    def dim_for(thr):
        idx = (cumulative >= thr).nonzero(as_tuple=True)[0]
        return int(idx[0]) + 1 if len(idx) > 0 else len(S)

    return {
        "feat_dim": M.shape[1],
        "full_rank": int((S > 1e-6).sum()),
        "stable_rank": float((eigvals.sum() / eigvals.max()).item()),
        "dim_90": dim_for(0.9),
        "dim_95": dim_for(0.95),
        "top16_energy": float(cumulative[15].item()) if len(cumulative) > 15 else float("nan"),
    }


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--tuned_ckpt", required=True)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_split", type=float, default=0.1)
    parser.add_argument("--out", default="analysis/block_feature_diff.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base model
    base_model = build_lora_classifier_qkv(args.base_ckpt).to(device)
    base_model.eval()

    # Tuned model
    tuned_model = build_lora_classifier_qkv(args.base_ckpt).to(device)
    tuned_model.load_state_dict(torch.load(args.tuned_ckpt, map_location=device), strict=False)
    tuned_model.eval()

    # Register hooks
    base_acts, base_handles = register_block_hooks(base_model)
    tuned_acts, tuned_handles = register_block_hooks(tuned_model)

    train_loader, _ = get_resisc45_dataloaders(
        train_split=args.train_split,
        batch_size=args.batch_size,
    )

    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            if i >= args.num_batches:
                break
            x = x.to(device)
            base_model(x)
            tuned_model(x)

    for h in base_handles + tuned_handles:
        h.remove()

    rows = []

    for layer in base_acts:
        if layer not in tuned_acts:
            continue

        F_pre = torch.cat(base_acts[layer], dim=0)
        F_post = torch.cat(tuned_acts[layer], dim=0)

        delta = F_post - F_pre

        fro_change = torch.norm(delta)
        rel_change = fro_change / (torch.norm(F_pre) + 1e-8)

        spec = compute_spectrum(delta)

        rows.append({
            "layer": layer,
            "fro_change": float(fro_change),
            "relative_change": float(rel_change),
            **spec
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Saved to:", args.out)


if __name__ == "__main__":
    main()