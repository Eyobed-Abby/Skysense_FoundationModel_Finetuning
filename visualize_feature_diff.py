import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

INPUT_CSV = "analysis/full_feature_diff.csv"
OUTPUT_DIR = "analysis/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------
# Helper: Extract Stage and Block
# ---------------------------------------------------------

def extract_stage_block(name):
    m = re.search(r"stages\.(\d+)\.blocks\.(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

df["stage"], df["block"] = zip(*df["layer"].map(extract_stage_block))

# Classify layer type
def layer_type(name):
    if "qkv" in name:
        return "QKV"
    elif name.endswith(".attn"):
        return "Attention"
    elif name.endswith(".ffn"):
        return "FFN"
    elif re.search(r"blocks\.\d+$", name):
        return "Block"
    else:
        return "Other"

df["type"] = df["layer"].apply(layer_type)

# ---------------------------------------------------------
# 1️⃣ Relative Change by Stage
# ---------------------------------------------------------

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="stage", y="relative_change", hue="type")
plt.title("Relative Feature Drift by Stage")
plt.savefig(f"{OUTPUT_DIR}/relative_change_by_stage.png")
plt.close()

# ---------------------------------------------------------
# 2️⃣ dim_90 Fraction Heatmap (Block-level)
# ---------------------------------------------------------

block_df = df[df["type"] == "Block"].copy()
block_df["dim_90_frac"] = block_df["dim_90"] / block_df["feat_dim"]

pivot = block_df.pivot(index="stage", columns="block", values="dim_90_frac")

plt.figure(figsize=(12,6))
sns.heatmap(pivot, annot=True, cmap="viridis")
plt.title("Intrinsic Dimension (dim_90_frac) Heatmap")
plt.savefig(f"{OUTPUT_DIR}/dim90_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 3️⃣ QKV vs Block Drift Comparison
# ---------------------------------------------------------

qkv_df = df[df["type"] == "QKV"]
block_df = df[df["type"] == "Block"]

plt.figure(figsize=(8,6))
plt.scatter(qkv_df["relative_change"], qkv_df["dim_90"], label="QKV")
plt.scatter(block_df["relative_change"], block_df["dim_90"], label="Block")
plt.xlabel("Relative Change")
plt.ylabel("dim_90")
plt.legend()
plt.title("QKV vs Block Drift")
plt.savefig(f"{OUTPUT_DIR}/qkv_vs_block.png")
plt.close()

# ---------------------------------------------------------
# 4️⃣ Effective Rank vs Depth
# ---------------------------------------------------------

plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="block", y="effective_rank", hue="stage", style="type")
plt.title("Effective Rank vs Block Depth")
plt.savefig(f"{OUTPUT_DIR}/effective_rank_depth.png")
plt.close()

# ---------------------------------------------------------
# 5️⃣ Suggested Rank Allocation (QKV Only)
# ---------------------------------------------------------

qkv_df["rank_suggestion"] = (
    qkv_df["dim_90"] / qkv_df["dim_90"].sum() * 64  # total rank budget 64 example
)

plt.figure(figsize=(12,6))
sns.barplot(data=qkv_df, x="layer", y="rank_suggestion")
plt.xticks(rotation=90)
plt.title("Suggested Dynamic LoRA Rank Allocation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rank_allocation.png")
plt.close()

print("All plots saved to:", OUTPUT_DIR)