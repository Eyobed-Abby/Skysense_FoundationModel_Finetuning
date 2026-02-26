import ast
import pandas as pd

# Path to your advanced feature diff CSV
CSV_PATH = "analysis/full_feature_diff2.csv"
OUT_SORTED = "analysis/full_feature_diff2_by_shape.csv"
OUT_SUMMARY = "analysis/full_feature_diff2_shape_summary.csv"

def parse_shape(x):
    """
    Convert weight_shape/feature_shape string like '(768, 768)' back to tuple.
    If it's None/NaN or invalid, return None.
    """
    if pd.isna(x):
        return None
    if isinstance(x, tuple):
        return x
    s = str(x).strip()
    if s in ("None", "", "nan"):
        return None
    try:
        return tuple(ast.literal_eval(s))
    except Exception:
        return None

def main():
    df = pd.read_csv(CSV_PATH)

    # Parse shapes into real tuples
    if "weight_shape" in df.columns:
        df["weight_shape_parsed"] = df["weight_shape"].apply(parse_shape)
    else:
        df["weight_shape_parsed"] = None

    if "feature_shape" in df.columns:
        df["feature_shape_parsed"] = df["feature_shape"].apply(parse_shape)
    else:
        df["feature_shape_parsed"] = None

    # Sort by weight_shape and then by layer name
    df_sorted = df.sort_values(
        by=["weight_shape_parsed", "feature_shape_parsed", "layer"]
    )

    # Save a CSV where layers with the same shape are grouped together
    df_sorted.to_csv(OUT_SORTED, index=False)
    print(f"[INFO] Saved shape-sorted table to: {OUT_SORTED}")

    # Filter to rows that actually have a weight matrix
    df_with_weight = df_sorted[~df_sorted["weight_shape_parsed"].isna()].copy()

    # Group by weight_shape and compute summary stats
    summary = (
        df_with_weight
        .groupby("weight_shape_parsed")
        .agg(
            num_layers=("layer", "count"),
            mean_dim90=("dim_90", "mean"),
            std_dim90=("dim_90", "std"),
            mean_dim90_frac=("dim_90_frac", "mean"),
            std_dim90_frac=("dim_90_frac", "std"),
            mean_stable_rank=("stable_rank", "mean"),
            std_stable_rank=("stable_rank", "std"),
            mean_rel_change=("rel_change", "mean"),
            std_rel_change=("rel_change", "std"),
        )
        .reset_index()
    )

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[INFO] Saved shape-wise summary to: {OUT_SUMMARY}")

    # Optionally, print summary to console for quick inspection
    print("\n=== Shape-wise Intrinsic Dimension Summary ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()