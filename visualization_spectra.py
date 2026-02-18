import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def make_plots(csv_path: str, out_dir: str = "analysis"):
    """
    Read weight_spectra.csv and produce several PNG plots that visualize
    intrinsic dimension and compressibility of each linear layer.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Derived ratios: how much of the full dimension you need
    df["ratio_50"] = df["dim_50"] / df["total_rank_max"]
    df["ratio_70"] = df["dim_70"] / df["total_rank_max"]
    df["ratio_90"] = df["dim_90"] / df["total_rank_max"]

    # 1) Histogram of stable_rank
    plt.figure()
    plt.hist(df["stable_rank"], bins=40)
    plt.xlabel("Stable rank")
    plt.ylabel("Number of layers")
    plt.title("Distribution of stable rank across layers")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stable_rank_hist.png"))
    plt.close()

    # 2) Histograms of ratio_50, ratio_70, ratio_90
    for ratio_col, thresh in [("ratio_50", "50"), ("ratio_70", "70"), ("ratio_90", "90")]:
        plt.figure()
        plt.hist(df[ratio_col], bins=40)
        plt.xlabel(f"Fraction of dimension for {thresh}% energy")
        plt.ylabel("Number of layers")
        plt.title(f"{thresh}% energy dimension as fraction of full rank")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{ratio_col}_hist.png"))
        plt.close()

    # 3) Scatter: total_rank_max vs dim_70
    plt.figure()
    plt.scatter(df["total_rank_max"], df["dim_70"])
    plt.xlabel("Total possible rank (min(m, n))")
    plt.ylabel("dim_70 (components for 70% energy)")
    plt.title("70% energy dimension vs. total possible rank")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dim70_vs_total_scatter.png"))
    plt.close()

    # 4) Boxplots of ratio_70 by group (qkv / ffn / proj / other), if available
    if "group" in df.columns:
        groups = ["qkv", "ffn", "proj", "other"]
        data = [df[df["group"] == g]["ratio_70"].dropna() for g in groups]

        filtered_groups = []
        filtered_data = []
        for g, d in zip(groups, data):
            if len(d) > 0:
                filtered_groups.append(g)
                filtered_data.append(d)

        if filtered_data:
            plt.figure()
            plt.boxplot(filtered_data, labels=filtered_groups)
            plt.ylabel("dim_70 / total_rank_max")
            plt.title("70% energy dimension fraction by layer group")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ratio70_by_group_boxplot.png"))
            plt.close()

    # 5) Scatter: stable_rank fraction vs ratio_70
    df["stable_fraction"] = df["stable_rank"] / df["total_rank_max"]
    plt.figure()
    plt.scatter(df["stable_fraction"], df["ratio_70"])
    plt.xlabel("Stable rank / total rank")
    plt.ylabel("dim_70 / total_rank_max")
    plt.title("Stable-rank fraction vs 70% energy fraction")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stable_vs_ratio70_scatter.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Make spectral plots from weight_spectra.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="analysis/weight_spectra.csv",
        help="Path to weight_spectra.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis",
        help="Directory to save PNG plots",
    )
    args = parser.parse_args()
    make_plots(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
