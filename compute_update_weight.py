import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_weight_spectrum(
    model,
    topk_list=[1, 2, 4, 8, 16, 32, 64],
    thresholds=[0.3, 0.5, 0.7, 0.9]
):
    results = []

    for name, param in model.named_parameters():
        if param.ndim == 2:  # Only linear layers
            W = param.detach().cpu()

            # SVD
            # S: singular values σ_i (sorted descending)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            # Eigenvalues of W^T W are λ_i = σ_i^2
            S_squared = S**2
            total_energy = S_squared.sum()

            # Avoid division by zero in case of a weird all-zero layer
            if total_energy == 0:
                # Fill with NaNs or zeros as you prefer
                layer_info = {
                    "layer": name,
                    "rank": len(S),
                    "total_energy": 0.0,
                }
                for k in topk_list:
                    if k <= len(S):
                        layer_info[f"top{k}_energy"] = 0.0
                for t in thresholds:
                    perc = int(t * 100)
                    layer_info[f"dim_{perc}"] = 0
                results.append(layer_info)
                continue

            # Cumulative eigenvalue ratio (explained variance)
            cumulative_energy = torch.cumsum(S_squared, dim=0) / total_energy

            layer_info = {
                "layer": name,
                "rank": len(S),
                "total_energy": total_energy.item()
            }

            # 1) Existing: for fixed k, what fraction of eigenvalue energy?
            for k in topk_list:
                if k <= len(S):
                    energy_k = S_squared[:k].sum() / total_energy
                    layer_info[f"top{k}_energy"] = energy_k.item()

            # 2) New: for fixed energy thresholds, what k do we need?
            #    Smallest k such that cumulative_energy[k-1] >= threshold
            for t in thresholds:
                perc = int(t * 100)  # e.g. 0.3 -> 30
                idx = (cumulative_energy >= t).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    k_needed = len(S)
                else:
                    k_needed = int(idx[0].item()) + 1  # +1 for 1-based k

                layer_info[f"dim_{perc}"] = k_needed

            results.append(layer_info)

    return pd.DataFrame(results)


if __name__ == "__main__":
    checkpoint_path = "skysense_model_backbone_hr.pth"
    model = torch.load(checkpoint_path, map_location="cpu")

    df = compute_weight_spectrum(model)
    df.to_csv("analysis/weight_spectra_analysis.csv", index=False)

    print(df.head())