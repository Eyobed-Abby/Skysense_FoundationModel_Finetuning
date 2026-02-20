import os
import csv
import argparse

import torch
import torch.nn as nn


def is_lora_module(name: str, module: nn.Module) -> bool:
    """
    Heuristic to skip LoRA layers:
    - Any module whose name contains 'lora'
    - Any module whose class name contains 'lora'
    """
    lname = name.lower()
    if "lora" in lname:
        return True
    clsname = module.__class__.__name__.lower()
    if "lora" in clsname:
        return True
    return False


def get_group_from_name(name: str) -> str:
    """
    Rough grouping of layers based on their names, like script B.
    """
    lname = name.lower()
    if "qkv" in lname:
        return "qkv"
    if "ffn" in lname or "feedforward" in lname:
        return "ffn"
    if "proj" in lname or "projection" in lname:
        return "proj"
    return "other"


def analyze_weight_matrix(
    W: torch.Tensor,
    thresholds=(0.3, 0.5, 0.7, 0.9, 0.95),
    topk_list=(4, 8, 16, 32, 64),
    rank_eps=1e-6,
):
    """
    Given a weight matrix W, compute:
      - full_rank (numerical rank)
      - stable_rank
      - frob_norm, spectral_norm
      - dim_X: minimal k s.t. sum_{i<=k} λ_i / sum_j λ_j >= X
      - dim_X_frac: dim_X / full_rank
      - topk_energy: energy captured by top-k eigenvalues

    Here λ_i = σ_i^2 are eigenvalues of W^T W.
    """
    # Ensure 2D
    W = W.detach().float().cpu()
    if W.ndim != 2:
        W = W.view(W.size(0), -1)

    # SVD (singular values σ_i)
    # Use full_matrices=False to keep it efficient.
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Guard against weird degenerate case
    if S.numel() == 0:
        return None

    # Sort singular values descending (should already be, but just to be safe)
    S = torch.sort(S, descending=True).values

    # Eigenvalues of W^T W
    eigenvals = S ** 2
    total_eig_sum = eigenvals.sum().item()

    if total_eig_sum <= 0:
        # All zeros – skip
        return None

    # Numerical full rank: how many singular values are "non-zero"
    full_rank = int((S > rank_eps).sum().item())
    if full_rank == 0:
        return None

    # Normalized eigenvalue ratios and cumulative energy
    eig_ratios = eigenvals / eigenvals.sum()
    cumulative_energy = torch.cumsum(eig_ratios, dim=0)

    # Stable rank = ||W||_F^2 / ||W||_2^2
    frob_sq = eigenvals.sum().item()
    spectral_sq = float(S[0].item() ** 2)
    stable_rank = frob_sq / (spectral_sq + 1e-12)

    metrics = {
        "full_rank": full_rank,
        "stable_rank": float(stable_rank),
        "frob_norm": float(frob_sq ** 0.5),
        "spectral_norm": float(S[0].item()),
    }

    # For each threshold: minimal k s.t. cumulative_energy[k-1] >= threshold
    for thr in thresholds:
        perc = int(thr * 100)
        idx = (cumulative_energy >= thr).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            k = full_rank
        else:
            k = int(idx[0].item()) + 1  # +1 because k is 1-based

        metrics[f"dim_{perc}"] = k
        metrics[f"dim_{perc}_frac"] = float(k / full_rank)

    # topk_energy: fraction of total eigenvalue mass explained by top-k
    for k in topk_list:
        if k <= eigenvals.numel():
            metrics[f"top{k}_energy"] = float(cumulative_energy[k - 1].item())
        else:
            metrics[f"top{k}_energy"] = float("nan")

    return metrics


def compute_weight_spectra(
    model: nn.Module,
    save_path: str,
    thresholds=(0.3, 0.5, 0.7, 0.9, 0.95),
    topk_list=(4, 8, 16, 32, 64),
    rank_eps=1e-6,
    min_rank_keep=10,
):
    """
    Iterate over nn.Linear layers in the model, skip LoRA layers and
    low-rank layers, compute spectral stats, and save to CSV.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    header = [
        "layer",
        "group",
        "shape",
        "full_rank",
        "stable_rank",
        "frob_norm",
        "spectral_norm",
    ]

    # Dim columns (k needed)
    for thr in thresholds:
        perc = int(thr * 100)
        header.append(f"dim_{perc}")       # number of eigenvalues (directions)
        header.append(f"dim_{perc}_frac")  # percentage of eigenvalues

    # Top-k energy columns
    for k in topk_list:
        header.append(f"top{k}_energy")

    rows = []

    for name, module in model.named_modules():
        # Only consider Linear layers
        if not isinstance(module, nn.Linear):
            continue

        # Skip LoRA layers
        if is_lora_module(name, module):
            continue

        weight = module.weight.data
        shape = tuple(weight.shape)

        metrics = analyze_weight_matrix(
            weight,
            thresholds=thresholds,
            topk_list=topk_list,
            rank_eps=rank_eps,
        )

        if metrics is None:
            continue

        # Skip layers whose numerical rank is below threshold
        if metrics["full_rank"] < min_rank_keep:
            continue

        group = get_group_from_name(name)

        row = [
            name,
            group,
            f"{shape[0]}x{shape[1]}",
            metrics["full_rank"],
            metrics["stable_rank"],
            metrics["frob_norm"],
            metrics["spectral_norm"],
        ]

        for thr in thresholds:
            perc = int(thr * 100)
            row.append(metrics[f"dim_{perc}"])
            row.append(metrics[f"dim_{perc}_frac"])

        for k in topk_list:
            row.append(metrics[f"top{k}_energy"])

        rows.append(row)

    # Write CSV
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[INFO] Saved spectral stats for {len(rows)} layers to: {save_path}")


if __name__ == "__main__":
    # IMPORTANT: adjust this import to match your repo structure
    from skysense_lora_classifier_qkv import build_lora_classifier_qkv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        default="skysense_model_backbone_hr.pth",
        help="Path to SkySense HR checkpoint",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="analysis/weight_spectra_merged.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Build full model (nn.Module) from checkpoint
    model = build_lora_classifier_qkv(args.ckpt)

    compute_weight_spectra(model, args.out)