import os
import re
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from resisc45_loader import get_resisc45_dataloaders
from skysense_lora_classifier_qkv import build_lora_classifier_qkv


# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------
# Utility: extract weight shape
# -------------------------------------------------------

def get_weight_shape(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, "weight"):
            return tuple(module.weight.shape)
    return None


# -------------------------------------------------------
# SVD Metrics
# -------------------------------------------------------

def compute_spectral_metrics(D):

    # Center
    D = D - D.mean(dim=0, keepdim=True)

    U, S, V = torch.linalg.svd(D, full_matrices=False)
    eigvals = S**2

    total_var = eigvals.sum()
    ratios = eigvals / (total_var + 1e-12)
    cumulative = torch.cumsum(ratios, dim=0)

    def dim_for(threshold):
        idx = (cumulative >= threshold).nonzero(as_tuple=True)[0]
        return int(idx[0]) + 1 if len(idx) > 0 else len(S)

    def topk_energy(k):
        if len(cumulative) >= k:
            return float(cumulative[k-1])
        else:
            return float("nan")

    stable_rank = float((eigvals.sum() / (eigvals.max() + 1e-12)))
    participation = float((eigvals.sum()**2 / (eigvals**2).sum()))

    return {
        "dim_90": dim_for(0.90),
        "dim_95": dim_for(0.95),
        "stable_rank": stable_rank,
        "participation_ratio": participation,
        "top4_energy": topk_energy(4),
        "top8_energy": topk_energy(8),
        "top16_energy": topk_energy(16),
        "top32_energy": topk_energy(32),
        "top64_energy": topk_energy(64),
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--tuned_ckpt", required=True)
    parser.add_argument("--train_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="analysis/feature_diff_advanced.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = build_lora_classifier_qkv(args.base_ckpt).to(device)
    tuned_model = build_lora_classifier_qkv(args.base_ckpt).to(device)
    tuned_model.load_state_dict(torch.load(args.tuned_ckpt, map_location=device), strict=False)

    base_model.eval()
    tuned_model.eval()

    activations_pre = {}
    activations_post = {}

    def hook_fn(name, store):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output

            if out.ndim == 4:
                out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
            elif out.ndim == 3:
                out = out.mean(dim=1)

            store.setdefault(name, []).append(out.detach().cpu())
        return hook

    pattern = re.compile(r"backbone\.stages\.\d+\.blocks\.\d+($|\.attn|\.ffn|\.attn\.w_msa\.qkv)")

    for name, module in base_model.named_modules():
        if pattern.search(name):
            module.register_forward_hook(hook_fn(name, activations_pre))

    for name, module in tuned_model.named_modules():
        if pattern.search(name):
            module.register_forward_hook(hook_fn(name, activations_post))

    loader, _ = get_resisc45_dataloaders(batch_size=args.batch_size, train_split=args.train_split)

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= args.num_batches:
                break
            x = x.to(device)
            base_model(x)
            tuned_model(x)

    rows = []

    for layer in activations_pre:

        if layer not in activations_post:
            continue

        A = torch.cat(activations_pre[layer], dim=0)
        B = torch.cat(activations_post[layer], dim=0)

        D = B - A

        weight_shape = get_weight_shape(base_model, layer)

        metrics = compute_spectral_metrics(D)

        fro_norm = float(torch.norm(D))
        rel_change = fro_norm / (float(torch.norm(A)) + 1e-12)

        rows.append({
            "layer": layer,
            "weight_shape": weight_shape,
            "feature_shape": tuple(D.shape),
            "feat_dim": D.shape[1],
            "num_samples": D.shape[0],
            "fro_norm": fro_norm,
            "rel_change": rel_change,
            **metrics
        })

    df = pd.DataFrame(rows)

    df["dim_90_frac"] = df["dim_90"] / df["feat_dim"]

    df.to_csv(args.out, index=False)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()