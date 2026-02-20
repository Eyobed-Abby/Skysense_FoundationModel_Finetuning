import os
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Helpers ----------

def ensure_outdir(path: str):
    if not path:
        path = "analysis"
    os.makedirs(path, exist_ok=True)
    return path


def parse_stage_block(name: str):
    """
    Parse Swin-style names like:
      base_model.model.backbone.stages.2.blocks.5.ffn...

    Returns (stage, block) as integers or (stage, None) if no block.
    """
    m = re.search(r"stages\.(\d+)\.blocks\.(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.search(r"stages\.(\d+)", name)
    if m2:
        return int(m2.group(1)), None
    return None, None


def load_csv_with_depth(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Add stage / block columns
    stages, blocks = [], []
    for name in df["layer"]:
        s, b = parse_stage_block(name)
        stages.append(s)
        blocks.append(b)
    df["stage"] = stages
    df["block"] = blocks

    # Stable-rank ratio if missing
    if "stable_rank_ratio" not in df.columns:
        df["stable_rank_ratio"] = df["stable_rank"] / df["full_rank"]

    # Only rows that belong to a block (inside transformer blocks)
    df_blocks = df[df["block"].notna()].copy()

    # Build a global depth index across all stages (stage 0 blocks first, then stage1, etc.)
    blocks_per_stage = (
        df_blocks.groupby("stage")["block"].max().astype(int) + 1
    )  # e.g. stage2 has 18 blocks
    offsets = {}
    cur = 0
    for s in sorted(blocks_per_stage.index):
        offsets[int(s)] = cur
        cur += blocks_per_stage[s]

    def global_depth(row):
        s = row["stage"]
        b = row["block"]
        if pd.isna(s) or pd.isna(b):
            return np.nan
        return offsets[int(s)] + int(b)

    df_blocks["global_depth"] = df_blocks.apply(global_depth, axis=1)

    return df, df_blocks


# ---------- PLOTS ----------

def plot_stage_group_heatmap(df_blocks: pd.DataFrame, outdir: str, col: str):
    """
    Heatmap: stage (0–3) vs group (qkv / proj / ffn / other),
    value = median of `col` (e.g. dim_90_frac).
    """
    pivot = (
        df_blocks
        .groupby(["stage", "group"])[col]
        .median()
        .reset_index()
        .pivot(index="group", columns="stage", values=col)
        .sort_index()
    )

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label=col)

    plt.xticks(
        ticks=np.arange(pivot.shape[1]),
        labels=[int(s) for s in pivot.columns],
    )
    plt.yticks(
        ticks=np.arange(pivot.shape[0]),
        labels=pivot.index,
    )
    plt.xlabel("Stage")
    plt.ylabel("Group")
    plt.title(f"Median {col} by stage and group")

    # Annotate cells with values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                plt.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=8, color="white"
                )

    plt.tight_layout()
    out_path = os.path.join(outdir, f"heatmap_stage_group_{col}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_block_depth_per_stage(df_blocks: pd.DataFrame, outdir: str, col: str):
    """
    For each stage, line plot:
      x = block index inside stage
      y = median of `col` per (stage, block, group)
    """
    stages = sorted(df_blocks["stage"].dropna().unique())
    groups = ["qkv", "proj", "ffn", "other"]

    for s in stages:
        sub = df_blocks[df_blocks["stage"] == s]
        if sub.empty:
            continue

        plt.figure(figsize=(7, 4))
        for g in groups:
            gsub = (
                sub[sub["group"] == g]
                .groupby("block")[col]
                .median()
                .reset_index()
                .sort_values("block")
            )
            if gsub.empty:
                continue
            plt.plot(gsub["block"], gsub[col], marker="o", label=g)

        plt.xlabel(f"Block index (stage {int(s)})")
        plt.ylabel(col)
        plt.title(f"{col} vs block depth in stage {int(s)}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(outdir, f"{col}_vs_block_stage{int(s)}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved {out_path}")


def plot_global_depth(df_blocks: pd.DataFrame, outdir: str, col: str):
    """
    Global depth index across all stages:
      x = global_depth (stage0 blocks then stage1, ...),
      y = median `col` per (group, global_depth).
    """
    groups = ["qkv", "proj", "ffn", "other"]
    plt.figure(figsize=(8, 4))
    for g in groups:
        sub = (
            df_blocks[df_blocks["group"] == g]
            .groupby("global_depth")[col]
            .median()
            .reset_index()
            .sort_values("global_depth")
        )
        if sub.empty:
            continue
        plt.plot(sub["global_depth"], sub[col], marker="o", label=g)

    plt.xlabel("Global depth index (across all stages)")
    plt.ylabel(col)
    plt.title(f"{col} vs global depth")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, f"{col}_vs_global_depth.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def get_topk_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("top") and c.endswith("_energy")]
    if not cols:
        return [], []
    def extract_k(col):
        return int(col.replace("top", "").replace("_energy", ""))
    cols = sorted(cols, key=extract_k)
    ks = [extract_k(c) for c in cols]
    return ks, cols


def plot_stage_topk_curves(df_blocks: pd.DataFrame, outdir: str):
    """
    For each stage and group, plot median top-k energy curve across blocks.
    Lets you see how quickly variance accumulates at each resolution.
    """
    ks, topk_cols = get_topk_columns(df_blocks)
    if not topk_cols:
        print("[WARN] No topk_energy columns, skipping stage top-k plots.")
        return

    stages = sorted(df_blocks["stage"].dropna().unique())
    groups = ["qkv", "proj", "ffn", "other"]

    for s in stages:
        plt.figure(figsize=(7, 4))
        sub_stage = df_blocks[df_blocks["stage"] == s]
        for g in groups:
            sub = sub_stage[sub_stage["group"] == g]
            if sub.empty:
                continue
            medians = [sub[c].median() for c in topk_cols]
            plt.plot(ks, medians, marker="o", label=g)

        plt.xlabel("k (top-k eigen-directions)")
        plt.ylabel("Median top-k energy (fraction of variance)")
        plt.title(f"Stage {int(s)}: median top-k energy curves by group")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(outdir, f"stage{int(s)}_topk_energy_curves.png")
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
        "--outdir",
        type=str,
        default="analysis",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    df, df_blocks = load_csv_with_depth(args.csv)

    # 1) Stage × group heatmaps
    plot_stage_group_heatmap(df_blocks, outdir, "dim_90_frac")
    plot_stage_group_heatmap(df_blocks, outdir, "stable_rank_ratio")

    # 2) Within-stage depth evolution
    plot_block_depth_per_stage(df_blocks, outdir, "dim_90_frac")
    plot_block_depth_per_stage(df_blocks, outdir, "stable_rank_ratio")

    # 3) Global depth index across all stages
    plot_global_depth(df_blocks, outdir, "dim_90_frac")
    plot_global_depth(df_blocks, outdir, "stable_rank_ratio")

    # 4) Stage-wise top-k energy curves
    plot_stage_topk_curves(df_blocks, outdir)


if __name__ == "__main__":
    main()