import os
import argparse
import re

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------- Helpers ----------

def ensure_outdir(path: str):
    if not path:
        path = "analysis"
    os.makedirs(path, exist_ok=True)
    return path


def parse_depth_from_name(name: str):
    """
    Heuristic: split 'layer' string on '.' and return the first token
    that is an integer. Works for names like:
        'blocks.0.attn.qkv'
        'layers.12.mlp.fc1'
    If no integer is found, return None.
    """
    tokens = name.split(".")
    for t in tokens:
        if t.isdigit():
            return int(t)
    # fallback: try regex for any digits
    m = re.search(r"\d+", name)
    if m:
        return int(m.group(0))
    return None


def load_csv_with_depth(csv_path: str):
    df = pd.read_csv(csv_path)

    # add depth column
    depths = []
    for name in df["layer"]:
        d = parse_depth_from_name(name)
        depths.append(d)
    df["depth"] = depths

    # stable_rank_ratio in case it's missing
    if "stable_rank_ratio" not in df.columns:
        df["stable_rank_ratio"] = df["stable_rank"] / df["full_rank"]

    return df


def plot_depth_vs_dim90_frac(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(8, 5))
    groups = sorted(df["group"].unique())

    for g in groups:
        sub = df[df["group"] == g]
        sub = sub.dropna(subset=["depth", "dim_90_frac"])
        if sub.empty:
            continue
        plt.scatter(
            sub["depth"],
            sub["dim_90_frac"],
            alpha=0.6,
            label=g,
            s=25,
        )

    plt.xlabel("Layer depth (parsed from name)")
    plt.ylabel("dim_90_frac (fraction of eigenvalues for 90% variance)")
    plt.title("Depth vs dim_90_frac")
    plt.grid(alpha=0.3)
    plt.legend(title="Group")
    plt.tight_layout()
    out_path = os.path.join(outdir, "depth_vs_dim90_frac.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_depth_vs_stable_ratio(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(8, 5))
    groups = sorted(df["group"].unique())

    for g in groups:
        sub = df[df["group"] == g]
        sub = sub.dropna(subset=["depth", "stable_rank_ratio"])
        if sub.empty:
            continue
        plt.scatter(
            sub["depth"],
            sub["stable_rank_ratio"],
            alpha=0.6,
            label=g,
            s=25,
        )

    plt.xlabel("Layer depth (parsed from name)")
    plt.ylabel("stable_rank / full_rank")
    plt.title("Depth vs stable-rank ratio (lower = more low-rank)")
    plt.grid(alpha=0.3)
    plt.legend(title="Group")
    plt.tight_layout()
    out_path = os.path.join(outdir, "depth_vs_stable_rank_ratio.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


# ---------- Cumulative energy plots (full spectrum) ----------

def build_module_dict(model: nn.Module):
    """
    Convenience dict: name -> module, using model.named_modules()
    """
    return {name: m for name, m in model.named_modules()}


def compute_eigen_spectrum(weight: torch.Tensor):
    """
    Given a 2D weight matrix, return the eigenvalues λ_i of W^T W
    (sorted descending) and the cumulative energy array.
    """
    W = weight.detach().float().cpu()
    if W.ndim != 2:
        W = W.view(W.size(0), -1)

    # SVD
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    if S.numel() == 0:
        return None, None

    S = torch.sort(S, descending=True).values
    eigenvals = (S ** 2).numpy()
    total = eigenvals.sum()
    if total <= 0:
        return None, None

    cum_energy = np.cumsum(eigenvals) / total
    return eigenvals, cum_energy


def select_depth_indices(depths):
    """
    Given a sorted list of unique depths, pick:
      - earliest
      - middle
      - last
    """
    if not depths:
        return []
    depths = sorted(depths)
    if len(depths) == 1:
        return [depths[0]]
    if len(depths) == 2:
        return [depths[0], depths[-1]]
    mid = depths[len(depths) // 2]
    return [depths[0], mid, depths[-1]]


def plot_cumulative_energy_for_group(df: pd.DataFrame,
                                     model: nn.Module,
                                     modules_dict: dict,
                                     group: str,
                                     outdir: str,
                                     max_k: int = 512):
    """
    For a given group (e.g. 'qkv', 'ffn', 'proj'), pick layers at
    early/mid/late depths and plot cumulative eigenvalue curves.
    """
    sub = df[(df["group"] == group) & (~df["depth"].isna())]
    if sub.empty:
        print(f"[WARN] No layers for group {group}, skipping cumulative plot.")
        return

    unique_depths = sorted(sub["depth"].dropna().unique().tolist())
    chosen_depths = select_depth_indices(unique_depths)

    if not chosen_depths:
        print(f"[WARN] Could not select depths for group {group}.")
        return

    plt.figure(figsize=(8, 5))

    for d in chosen_depths:
        # pick first layer at this depth
        row = sub[sub["depth"] == d].iloc[0]
        layer_name = row["layer"]

        if layer_name not in modules_dict:
            print(f"[WARN] Layer {layer_name} not found in model.named_modules(), skipping.")
            continue

        module = modules_dict[layer_name]
        if not isinstance(module, nn.Linear):
            # If weight is not from Linear, try to handle generically
            if not hasattr(module, "weight"):
                print(f"[WARN] Module {layer_name} has no weight, skipping.")
                continue
            weight = module.weight.data
        else:
            weight = module.weight.data

        eigenvals, cum_energy = compute_eigen_spectrum(weight)
        if eigenvals is None:
            continue

        Ks = np.arange(1, len(eigenvals) + 1)
        if max_k is not None:
            # cap to max_k for readability
            max_k_eff = min(max_k, len(eigenvals))
            Ks = Ks[:max_k_eff]
            cum_energy = cum_energy[:max_k_eff]

        label = f"depth {d} ({layer_name})"
        plt.plot(Ks, cum_energy, label=label)

    plt.xlabel("k (top-k eigen-directions)")
    plt.ylabel("Cumulative variance explained")
    plt.title(f"Cumulative eigenvalue energy curves for group '{group}'")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(outdir, f"cumulative_energy_{group}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="analysis/weight_spectra_merged.csv",
        help="Path to weight_spectra_merged.csv",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to SkySense HR checkpoint (same as for spectral analysis)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="analysis",
        help="Directory to save depth-wise plots",
    )
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)

    # 1) Load CSV and add depth column
    df = load_csv_with_depth(args.csv)

    # 2) Depth-aware plots from summary metrics
    plot_depth_vs_dim90_frac(df, outdir)
    plot_depth_vs_stable_ratio(df, outdir)

    # 3) Build model to recompute full spectra for selected layers
    from skysense_lora_classifier_qkv import build_lora_classifier_qkv

    model = build_lora_classifier_qkv(args.ckpt)
    modules_dict = build_module_dict(model)

    # 4) For each main group, create cumulative energy curves
    for group in ["qkv", "ffn", "proj"]:
        plot_cumulative_energy_for_group(df, model, modules_dict, group, outdir)


if __name__ == "__main__":
    main()