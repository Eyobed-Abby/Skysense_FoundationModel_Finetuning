import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_outdir(path: str):
    if path is None or path == "":
        path = "analysis"
    os.makedirs(path, exist_ok=True)
    return path


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Derived columns
    df["stable_rank_ratio"] = df["stable_rank"] / df["full_rank"]
    # Just to be safe, clip fractions into [0,1]
    for col in ["dim_30_frac", "dim_50_frac", "dim_70_frac", "dim_90_frac", "dim_95_frac"]:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    return df


def plot_stable_rank_ratio_by_group(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(8, 5))
    groups = sorted(df["group"].unique())

    data = [df.loc[df["group"] == g, "stable_rank_ratio"].values for g in groups]

    plt.boxplot(data, labels=groups, showfliers=False)
    plt.ylabel("stable_rank / full_rank")
    plt.xlabel("Layer group")
    plt.title("Stable-rank ratio by group\n(lower = more low-rank)")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(outdir, "stable_rank_ratio_by_group.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_dim_frac_by_group(df: pd.DataFrame, outdir: str, dim_col: str):
    """
    dim_col is one of: 'dim_30_frac', 'dim_50_frac', 'dim_70_frac',
    'dim_90_frac', 'dim_95_frac'
    """
    if dim_col not in df.columns:
        print(f"[WARN] Column {dim_col} not in DataFrame, skipping.")
        return

    plt.figure(figsize=(8, 5))
    groups = sorted(df["group"].unique())

    data = [df.loc[df["group"] == g, dim_col].values for g in groups]

    plt.boxplot(data, labels=groups, showfliers=False)
    plt.ylabel(f"{dim_col} (fraction of eigenvalues)")
    plt.xlabel("Layer group")
    thr = dim_col.split("_")[1]  # e.g. '90' from 'dim_90_frac'
    plt.title(f"Fraction of eigenvalues needed for {thr}% variance\n({dim_col})")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(outdir, f"{dim_col}_by_group.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_topk_energy_median_by_group(df: pd.DataFrame, outdir: str):
    """
    For each group, compute median topk_energy curves over k in {4, 8, 16, 32, 64}
    and plot them on the same figure.
    """
    topk_cols = [c for c in df.columns if c.startswith("top") and c.endswith("_energy")]
    if not topk_cols:
        print("[WARN] No topk_energy columns found, skipping topk curves.")
        return

    # Sort columns by k
    def extract_k(col):
        # 'top4_energy' -> 4
        return int(col.replace("top", "").replace("_energy", ""))

    topk_cols = sorted(topk_cols, key=extract_k)
    ks = [extract_k(c) for c in topk_cols]

    groups = sorted(df["group"].unique())
    plt.figure(figsize=(8, 5))

    for g in groups:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        medians = [sub[c].median() for c in topk_cols]
        plt.plot(ks, medians, marker="o", label=g)

    plt.xlabel("k (top-k eigen-directions)")
    plt.ylabel("Median top-k energy (fraction of variance)")
    plt.title("Median top-k eigenvalue energy curves per group")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(title="Group")
    plt.tight_layout()
    out_path = os.path.join(outdir, "topk_energy_median_by_group.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_scatter_full_vs_stable(df: pd.DataFrame, outdir: str):
    """
    full_rank vs stable_rank, colored by group.
    Shows how 'effective' dimensionality compares to maximum rank.
    """
    plt.figure(figsize=(7, 6))
    groups = sorted(df["group"].unique())

    for g in groups:
        sub = df[df["group"] == g]
        plt.scatter(
            sub["full_rank"],
            sub["stable_rank"],
            alpha=0.6,
            label=g,
            s=25,
        )

    max_rank = df["full_rank"].max()
    plt.plot([0, max_rank], [0, max_rank], "k--", linewidth=1, alpha=0.5)

    plt.xlabel("full_rank")
    plt.ylabel("stable_rank")
    plt.title("Stable rank vs full rank\n(dashed line = stable == full)")
    plt.grid(alpha=0.3)
    plt.legend(title="Group")
    plt.tight_layout()
    out_path = os.path.join(outdir, "stable_vs_full_rank_scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_scatter_full_vs_dim90(df: pd.DataFrame, outdir: str):
    """
    Scatter full_rank vs dim_90, colored by group.
    Highlights how many directions are needed for 90% variance
    relative to the maximum possible rank.
    """
    if "dim_90" not in df.columns:
        print("[WARN] dim_90 not found, skipping.")
        return

    plt.figure(figsize=(7, 6))
    groups = sorted(df["group"].unique())

    for g in groups:
        sub = df[df["group"] == g]
        plt.scatter(
            sub["full_rank"],
            sub["dim_90"],
            alpha=0.6,
            label=g,
            s=25,
        )

    max_rank = df["full_rank"].max()
    plt.plot([0, max_rank], [0, max_rank], "k--", linewidth=1, alpha=0.5)

    plt.xlabel("full_rank")
    plt.ylabel("dim_90 (directions for 90% variance)")
    plt.title("dim_90 vs full_rank\n(dashed line = dim_90 == full_rank)")
    plt.grid(alpha=0.3)
    plt.legend(title="Group")
    plt.tight_layout()
    out_path = os.path.join(outdir, "dim90_vs_full_rank_scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def print_summary_stats(df: pd.DataFrame):
    """
    Print some useful summary stats to the console.
    """
    groups = sorted(df["group"].unique())
    print("\n===== SUMMARY STATS BY GROUP =====")
    for g in groups:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        print(f"\nGroup: {g}")
        print(f"  #layers: {len(sub)}")
        print(f"  mean stable_rank:        {sub['stable_rank'].mean():.2f}")
        print(f"  mean full_rank:          {sub['full_rank'].mean():.2f}")
        print(f"  mean stable_rank_ratio:  {sub['stable_rank_ratio'].mean():.4f}")
        if 'dim_90_frac' in sub.columns:
            print(f"  mean dim_90_frac:        {sub['dim_90_frac'].mean():.4f}")
        if 'dim_95_frac' in sub.columns:
            print(f"  mean dim_95_frac:        {sub['dim_95_frac'].mean():.4f}")
        if 'top16_energy' in sub.columns:
            print(f"  mean top16_energy:       {sub['top16_energy'].mean():.4f}")
        if 'top32_energy' in sub.columns:
            print(f"  mean top32_energy:       {sub['top32_energy'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="analysis/weight_spectra_merged.csv",
        help="Path to weight_spectra_merged.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="analysis",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    df = load_data(args.csv)

    # Print some text insights
    print_summary_stats(df)

    # Plots
    plot_stable_rank_ratio_by_group(df, outdir)
    plot_dim_frac_by_group(df, outdir, "dim_90_frac")
    plot_dim_frac_by_group(df, outdir, "dim_95_frac")
    plot_topk_energy_median_by_group(df, outdir)
    plot_scatter_full_vs_stable(df, outdir)
    plot_scatter_full_vs_dim90(df, outdir)


if __name__ == "__main__":
    main()