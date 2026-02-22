import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from resisc45_loader import get_resisc45_dataloaders
from skysense_lora_classifier_qkv import (
    SkySenseClassifier,
    load_skysense_backbone,
    build_lora_classifier_qkv,
)


# ---------- Spectrum utils (same as before, but reused per-layer) ----------

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


# ---------- Hook machinery for per-layer feature capture ----------

def is_lora_module(name: str, module: nn.Module) -> bool:
    lname = name.lower()
    if "lora" in lname:
        return True
    clsname = module.__class__.__name__.lower()
    if "lora" in clsname:
        return True
    return False


def layer_filter(name: str, module: nn.Module) -> bool:
    """
    Decide which layers to hook.

    For now:
      - only backbone layers
      - only nn.Linear
      - skip LoRA modules
    """
    if is_lora_module(name, module):
        return False
    if "backbone" not in name:
        return False
    if not isinstance(module, nn.Linear):
        return False
    return True


def register_hooks(model: nn.Module):
    """
    Register forward hooks on selected layers.

    Returns:
      activations: dict name -> list of tensors [B, C]
      handles: list of hook handles (to remove later)
    """
    activations = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            out = output
            # Unpack tuples if needed
            if isinstance(out, tuple):
                out = out[0]
            # For safety: if more than 2D (e.g. [B, C, H, W]), global pool
            if out.ndim > 2:
                out = nn.functional.adaptive_avg_pool2d(out, 1)
                out = out.squeeze(-1).squeeze(-1)  # [B, C]
            elif out.ndim == 1:
                out = out.unsqueeze(0)  # [1, C]
            out = out.detach().cpu()
            activations.setdefault(name, []).append(out)
        return hook

    for name, module in model.named_modules():
        if layer_filter(name, module):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    return activations, handles


# ---------- Main logic ----------

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
        help="Path to fine-tuned LoRA classifier checkpoint (e.g. jobs/checkpoints/sanity_model.pth)",
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
        default="analysis/feature_diff_layer_spectrum.csv",
        help="Where to save layer-wise spectrum metrics.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Build base (pre-finetune) classifier without LoRA ---
    backbone_base = load_skysense_backbone(args.base_ckpt)
    base_model = SkySenseClassifier(backbone_base).to(device)
    base_model.eval()

    # --- Build LoRA classifier and load fine-tuned weights ---
    tuned_model = build_lora_classifier_qkv(args.base_ckpt).to(device)
    state = torch.load(args.tuned_ckpt, map_location=device)
    tuned_model.load_state_dict(state)
    tuned_model.eval()

    # --- Register hooks ---
    print("[INFO] Registering hooks on base model...")
    base_acts, base_handles = register_hooks(base_model)

    print("[INFO] Registering hooks on tuned model...")
    tuned_acts, tuned_handles = register_hooks(tuned_model)

    # --- Data loader ---
    train_loader, _ = get_resisc45_dataloaders(
        train_split=args.train_split,
        batch_size=args.batch_size,
    )

    # --- Run a few batches and collect activations ---
    print(f"[INFO] Collecting activations from {args.num_batches} batches...")
    base_model.eval()
    tuned_model.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            if i >= args.num_batches:
                break
            x = x.to(device)
            # Forward passes (hooks will fill the dicts)
            base_model(x)
            tuned_model(x)

    # Remove hooks
    for h in base_handles + tuned_handles:
        h.remove()

    print("[INFO] Finished collecting activations. Computing layer-wise spectra...")

    rows = []

    # Intersection of layers present in both dictionaries
    layer_names = sorted(set(base_acts.keys()).intersection(set(tuned_acts.keys())))

    for name in layer_names:
        A_base_list = base_acts[name]
        A_tuned_list = tuned_acts[name]
        if not A_base_list or not A_tuned_list:
            continue

        A_base = torch.cat(A_base_list, dim=0)   # [N, C_l]
        A_tuned = torch.cat(A_tuned_list, dim=0) # [N, C_l]

        if A_base.shape != A_tuned.shape:
            print(f"[WARN] Shape mismatch for layer {name}: "
                  f"base {tuple(A_base.shape)} vs tuned {tuple(A_tuned.shape)}, skipping.")
            continue

        D = A_tuned - A_base  # [N, C_l]

        metrics = analyze_matrix_spectrum(D)
        if metrics is None:
            print(f"[WARN] Degenerate spectrum for layer {name}, skipping.")
            continue

        metrics["layer"] = name
        metrics["num_samples"] = int(D.shape[0])
        metrics["feat_dim"] = int(D.shape[1])

        rows.append(metrics)

    if not rows:
        raise RuntimeError("No valid layers analyzed. Check layer_filter / hooks.")

    # Save to CSV
    df = pd.DataFrame(rows)
    # Reorder columns to put layer info first
    cols = ["layer", "num_samples", "feat_dim",
            "full_rank", "stable_rank", "frob_norm", "spectral_norm"]
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]

    df.to_csv(args.out, index=False)
    print(f"[INFO] Saved layer-wise feature-diff spectrum to {args.out}")

    # Optional: quick text summary
    print("\n==== SAMPLE OF RESULTS ====")
    print(df.head())


if __name__ == "__main__":
    main()