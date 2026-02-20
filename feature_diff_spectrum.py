import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from resisc45_loader import get_resisc45_dataloaders           # :contentReference[oaicite:2]{index=2}
from skysense_lora_classifier_qkv import (                      # :contentReference[oaicite:3]{index=3}
    SkySenseClassifier,
    load_skysense_backbone,
    build_lora_classifier_qkv,
)


# ---------- Spectrum utils (same logic as weight spectra) ----------

def analyze_matrix_spectrum(
    M: torch.Tensor,
    thresholds=(0.3, 0.5, 0.7, 0.9, 0.95),
    topk_list=(4, 8, 16, 32, 64),
    rank_eps=1e-6,
):
    """
    Given a 2D matrix M (e.g., [num_samples, feature_dim]), compute:
      - full_rank (numerical rank)
      - stable_rank
      - frob_norm, spectral_norm
      - dim_X: minimal k s.t. sum_{i<=k} λ_i / sum_j λ_j >= X
      - dim_X_frac: dim_X / full_rank
      - topk_energy: cumulative energy at top-k eigenvalues

    Here λ_i are eigenvalues of M^T M (i.e., squared singular values).
    """
    M = M.detach().float().cpu()
    if M.ndim != 2:
        M = M.view(M.size(0), -1)

    # SVD
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    if S.numel() == 0:
        return None

    # Ensure descending
    S = torch.sort(S, descending=True).values

    eigenvals = S ** 2
    total = eigenvals.sum().item()
    if total <= 0:
        return None

    # Numerical rank
    full_rank = int((S > rank_eps).sum().item())
    if full_rank == 0:
        return None

    eig_ratios = eigenvals / eigenvals.sum()
    cumulative = torch.cumsum(eig_ratios, dim=0)

    # Stable rank
    frob_sq = eigenvals.sum().item()
    spectral_sq = float(S[0].item() ** 2)
    stable_rank = frob_sq / (spectral_sq + 1e-12)

    metrics = {
        "full_rank": full_rank,
        "stable_rank": float(stable_rank),
        "frob_norm": float(frob_sq ** 0.5),
        "spectral_norm": float(S[0].item()),
    }

    # Threshold-based intrinsic dims
    for thr in thresholds:
        perc = int(thr * 100)
        idx = (cumulative >= thr).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            k = full_rank
        else:
            k = int(idx[0].item()) + 1
        metrics[f"dim_{perc}"] = k
        metrics[f"dim_{perc}_frac"] = float(k / full_rank)

    # top-k energies
    for k in topk_list:
        if k <= eigenvals.numel():
            metrics[f"top{k}_energy"] = float(cumulative[k - 1].item())
        else:
            metrics[f"top{k}_energy"] = float("nan")

    return metrics


# ---------- Feature extraction ----------

def extract_backbone_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Replicates the feature logic in SkySenseClassifier.forward but returns
    the pooled backbone features BEFORE the final classifier. :contentReference[oaicite:4]{index=4}
    """
    feats = model.backbone(x)
    if isinstance(feats, tuple):
        feats = feats[0]
    if feats.ndim == 4:
        pooled = nn.functional.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
    elif feats.ndim == 2:
        pooled = feats
    else:
        raise ValueError(f"Unexpected feature shape: {feats.shape}")
    return pooled  # [B, C]


def collect_feature_differences(
    base_model: nn.Module,
    tuned_model: nn.Module,
    dataloader,
    device,
    num_batches: int = 10,
):
    """
    Run a few mini-batches and collect Δf = f_tuned(x) - f_base(x)
    at the backbone-pooled feature level.

    Returns a [N, C] tensor where N = total samples processed.
    """
    base_model.eval()
    tuned_model.eval()

    diffs = []
    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            if count >= num_batches:
                break
            x = x.to(device)

            f_base = extract_backbone_features(base_model, x)   # [B, C]
            f_tuned = extract_backbone_features(tuned_model, x) # [B, C]

            delta = (f_tuned - f_base).cpu()
            diffs.append(delta)

            count += 1

    if not diffs:
        raise RuntimeError("No batches processed – check num_batches / dataloader.")
    return torch.cat(diffs, dim=0)  # [N, C]


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_ckpt",
        type=str,
        required=True,
        help="Path to base SkySense backbone checkpoint (skysense_model_backbone_hr.pth)",
    )
    parser.add_argument(
        "--tuned_ckpt",
        type=str,
        required=True,
        help="Path to fine-tuned LoRA classifier checkpoint (e.g. checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.1,
        help="Train split used for RESISC45 loader (must match training).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature sampling.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to sample for feature differences.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="analysis/feature_diff_spectrum.csv",
        help="Where to save spectrum metrics.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Build base (pre-finetune) classifier without LoRA ---
    backbone_base = load_skysense_backbone(args.base_ckpt)  # :contentReference[oaicite:5]{index=5}
    base_model = SkySenseClassifier(backbone_base).to(device)
    base_model.eval()

    # --- Build LoRA classifier and load fine-tuned weights ---
    tuned_model = build_lora_classifier_qkv(args.base_ckpt).to(device)  # same arch :contentReference[oaicite:6]{index=6}
    state = torch.load(args.tuned_ckpt, map_location=device)
    tuned_model.load_state_dict(state)
    tuned_model.eval()

    # --- Data loader (we only need a few batches) ---
    train_loader, _ = get_resisc45_dataloaders(
        train_split=args.train_split,
        batch_size=args.batch_size,
    )

    print(f"[INFO] Collecting feature differences from {args.num_batches} batches...")
    feature_diffs = collect_feature_differences(
        base_model, tuned_model, train_loader, device, num_batches=args.num_batches
    )
    print(f"[INFO] Collected feature differences of shape: {tuple(feature_diffs.shape)}")

    # --- Spectrum analysis ---
    metrics = analyze_matrix_spectrum(feature_diffs)

    if metrics is None:
        raise RuntimeError("Spectrum analysis failed (degenerate matrix).")

    # Save metrics as one-row CSV
    df = pd.DataFrame([metrics])
    df.to_csv(args.out, index=False)
    print(f"[INFO] Saved feature-diff spectrum metrics to {args.out}")

    # Also print a quick summary
    print("\n==== FEATURE DIFFERENCE SPECTRUM SUMMARY ====")
    for k, v in metrics.items():
        if k.startswith("dim_") or k.startswith("top"):
            print(f"{k:15s}: {v:.4f}" if isinstance(v, float) else f"{k:15s}: {v}")
        else:
            print(f"{k:15s}: {v}")


if __name__ == "__main__":
    main()