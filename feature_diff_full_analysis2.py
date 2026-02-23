import os
import re
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from resisc45_loader import get_resisc45_dataloaders
from skysense_lora_classifier_qkv import build_lora_classifier_qkv


# ==========================================================
# 1. Reproducibility
# ==========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# 2. Hook Definitions
# ==========================================================

def match_layers(name):
    patterns = [
        r"backbone\.stages\.\d+\.blocks\.\d+$",                # block
        r"backbone\.stages\.\d+\.blocks\.\d+\.attn$",          # attention
        r"backbone\.stages\.\d+\.blocks\.\d+\.attn\.w_msa\.qkv$",  # qkv
        r"backbone\.stages\.\d+\.blocks\.\d+\.ffn$",           # ffn
    ]
    return any(re.search(p, name) for p in patterns)


def register_hooks(model):
    activations = {}
    handles = []

    def hook_fn(name):
        def hook(module, input, output):
            out = output
            if isinstance(out, tuple):
                out = out[0]

            if out.ndim == 4:
                out = nn.functional.adaptive_avg_pool2d(out, 1)
                out = out.squeeze(-1).squeeze(-1)
            elif out.ndim == 3:
                out = out.mean(dim=1)

            out = out.detach().cpu()
            activations.setdefault(name, []).append(out)
        return hook

    for name, module in model.named_modules():
        if match_layers(name):
            handles.append(module.register_forward_hook(hook_fn(name)))

    return activations, handles


# ==========================================================
# 3. Spectrum Metrics
# ==========================================================

def compute_spectrum(D):

    D = D - D.mean(dim=0, keepdim=True)

    U, S, V = torch.linalg.svd(D, full_matrices=False)
    eigvals = S**2
    total = eigvals.sum()

    ratios = eigvals / total
    cumulative = torch.cumsum(ratios, dim=0)

    def dim_for(thr):
        idx = (cumulative >= thr).nonzero(as_tuple=True)[0]
        return int(idx[0]) + 1 if len(idx) > 0 else len(S)

    stable_rank = float((eigvals.sum() / eigvals.max()).item())
    participation_ratio = float((eigvals.sum()**2 / (eigvals**2).sum()).item())

    p = eigvals / eigvals.sum()
    entropy_rank = float(torch.exp(-(p * torch.log(p + 1e-8)).sum()).item())

    return {
        "feat_dim": D.shape[1],
        "full_rank": int((S > 1e-6).sum()),
        "stable_rank": stable_rank,
        "participation_ratio": participation_ratio,
        "effective_rank": entropy_rank,
        "dim_90": dim_for(0.9),
        "dim_95": dim_for(0.95),
    }


# ==========================================================
# 4. Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--tuned_ckpt", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--train_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="analysis/full_feature_diff.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = build_lora_classifier_qkv(args.base_ckpt).to(device)
    tuned_model = build_lora_classifier_qkv(args.base_ckpt).to(device)

    tuned_model.load_state_dict(
        torch.load(args.tuned_ckpt, map_location=device),
        strict=False
    )

    base_model.eval()
    tuned_model.eval()

    base_acts, base_handles = register_hooks(base_model)
    tuned_acts, tuned_handles = register_hooks(tuned_model)

    train_loader, _ = get_resisc45_dataloaders(
        train_split=args.train_split,
        batch_size=args.batch_size
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

        D = F_post - F_pre

        fro = torch.norm(D)
        rel = fro / (torch.norm(F_pre) + 1e-8)

        spec = compute_spectrum(D)

        rows.append({
            "layer": layer,
            "fro_change": float(fro),
            "relative_change": float(rel),
            **spec
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()