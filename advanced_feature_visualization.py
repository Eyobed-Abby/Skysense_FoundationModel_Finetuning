import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "analysis/full_feature_diff.csv"
OUTPUT_DIR = "analysis/advanced_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------
# Extract Stage and Block
# ---------------------------------------------------------
def extract_stage_block(name):
    m = re.search(r"stages\.(\d+)\.blocks\.(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

df["stage"], df["block"] = zip(*df["layer"].map(extract_stage_block))

# ---------------------------------------------------------
# Create Layer Type Column
# ---------------------------------------------------------
def classify_layer(name):
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

df["type"] = df["layer"].apply(classify_layer)

# ---------------------------------------------------------
# 1️⃣ Participation Ratio vs Relative Change
# ---------------------------------------------------------
if "participation_ratio" in df.columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="relative_change", y="participation_ratio", hue="type")
    plt.title("Participation Ratio vs Relative Change")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/participation_vs_change.png")
    plt.close()

# ---------------------------------------------------------
# 2️⃣ Normalized Intrinsic Dimension vs Depth (Block Only)
# ---------------------------------------------------------
block_df = df[df["type"] == "Block"].copy()

if len(block_df) > 0:
    block_df["dim_90_frac"] = block_df["dim_90"] / block_df["feat_dim"]

    plt.figure(figsize=(10,6))
    sns.lineplot(data=block_df, x="block", y="dim_90_frac", hue="stage", marker="o")
    plt.title("Normalized Intrinsic Dimension Across Depth (Blocks)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/normalized_depth_trend.png")
    plt.close()

# ---------------------------------------------------------
# 3️⃣ Relative Change vs dim_90 (Colored by Type)
# ---------------------------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="relative_change", y="dim_90", hue="type")
plt.title("Relative Change vs Intrinsic Dimension")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/change_vs_dim90.png")
plt.close()

# ---------------------------------------------------------
# 4️⃣ Stage-wise Mean Drift
# ---------------------------------------------------------
plt.figure(figsize=(8,6))
sns.barplot(data=df, x="stage", y="relative_change", hue="type")
plt.title("Mean Relative Drift by Stage")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mean_drift_by_stage.png")
plt.close()

# ---------------------------------------------------------
# 5️⃣ dim_90 Fraction Distribution
# ---------------------------------------------------------
df["dim_90_frac"] = df["dim_90"] / df["feat_dim"]

plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="type", y="dim_90_frac")
plt.title("Intrinsic Dimension Fraction by Layer Type")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dim90_fraction_by_type.png")
plt.close()

print("Advanced plots saved to:", OUTPUT_DIR)