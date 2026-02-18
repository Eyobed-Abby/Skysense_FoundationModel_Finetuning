import os
import csv
import torch
import torch.nn as nn


def spectral_energy_analysis(W: torch.Tensor):
    """
    Compute singular values and intrinsic-dimension style metrics for a matrix W.

    Returns:
        results: dict with stable_rank, energy dims, etc.
        s: 1D tensor of singular values (sorted descending).
        cumulative_energy: 1D tensor of cumulative energy ratios.
    """
    # Compute singular values (1D tensor, usually sorted descending already)
    s = torch.linalg.svdvals(W)

    # Sort explicitly just to be safe
    s, _ = torch.sort(s, descending=True)

    # Energy = squared singular values
    energy = s ** 2
    total_energy = energy.sum()

    # Handle degenerate case (all zeros)
    if total_energy == 0:
        cumulative_energy = torch.zeros_like(energy)
        results = {
            "stable_rank": 0.0,
            "dim_30": 0,
            "dim_50": 0,
            "dim_70": 0,
            "dim_90": 0,
            "dim_95": 0,
            "total_rank_max": int(s.numel()),
            "frob_norm": 0.0,
            "spectral_norm": 0.0,
        }
        return results, s, cumulative_energy

    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy

    def k_for_threshold(threshold: float) -> int:
        """
        Smallest k such that cumulative energy >= threshold.
        Returns 0 if even the first singular value doesn't reach threshold
        (which won't happen for reasonable thresholds and nonzero matrices).
        """
        idx = (cumulative_energy >= threshold).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return int(s.numel())
        return int(idx[0].item()) + 1  # +1 because k is count, not index

    stable_rank = (energy.sum() / (s.max() ** 2)).item()
    frob_norm = torch.sqrt(total_energy).item()
    spectral_norm = s.max().item()

    results = {
        "stable_rank": stable_rank,
        "dim_30": k_for_threshold(0.30),
        "dim_50": k_for_threshold(0.50),
        "dim_70": k_for_threshold(0.70),
        "dim_90": k_for_threshold(0.90),
        "dim_95": k_for_threshold(0.95),
        "total_rank_max": int(s.numel()),
        "frob_norm": frob_norm,
        "spectral_norm": spectral_norm,
    }

    return results, s, cumulative_energy


def compute_spectral_stats(model: nn.Module, save_path: str):
    """
    Iterate over all nn.Linear layers in the model, compute spectral stats,
    and save them to CSV.
    """
    print("\nSpectral Energy Report for Linear Layers\n")

    # CSV header
    rows = [(
        "layer",
        "group",
        "shape",
        "stable_rank",
        "dim_30",
        "dim_50",
        "dim_70",
        "dim_90",
        "dim_95",
        "total_rank_max",
        "frob_norm",
        "spectral_norm",
    )]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get weight as a 2D tensor on CPU
            weight = module.weight.data.detach().cpu()
            shape = tuple(weight.shape)

            # Group label (same logic you used for qkv/ffn/proj/other)
            if ".qkv" in name:
                group = "qkv"
            elif ".ffn" in name:
                group = "ffn"
            elif ".proj" in name:
                group = "proj"
            else:
                group = "other"

            # Compute spectral stats
            results, s, cum_energy = spectral_energy_analysis(weight)

            print(
                f"{name:60} | "
                f"shape: {str(shape):>15} | "
                f"stable_rank: {results['stable_rank']:.2f} | "
                f"dim_50: {results['dim_50']:4d} | "
                f"dim_90: {results['dim_90']:4d}"
            )

            rows.append((
                name,
                group,
                str(shape),
                results["stable_rank"],
                results["dim_30"],
                results["dim_50"],
                results["dim_70"],
                results["dim_90"],
                results["dim_95"],
                results["total_rank_max"],
                results["frob_norm"],
                results["spectral_norm"],
            ))

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save CSV
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nSaved spectral report to {save_path}")


if __name__ == "__main__":
    import argparse
    from skysense_lora_classifier_qkv import build_lora_classifier_qkv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to SkySense HR checkpoint",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="analysis/weight_spectra.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Build model from checkpoint (same pattern as your original script)
    model = build_lora_classifier_qkv(args.ckpt)

    # Run spectral analysis
    compute_spectral_stats(model, args.out)
