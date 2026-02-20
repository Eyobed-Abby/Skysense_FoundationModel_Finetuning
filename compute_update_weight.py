import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_weight_spectrum(model, topk_list=[1, 2, 4, 8, 16, 32, 64]):
    results = []

    for name, param in model.named_parameters():
        if param.ndim == 2:  # Only linear layers
            W = param.detach().cpu()

            # SVD
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            S_squared = S**2
            total_energy = S_squared.sum()

            layer_info = {
                "layer": name,
                "rank": len(S),
                "total_energy": total_energy.item()
            }

            for k in topk_list:
                if k <= len(S):
                    energy_k = S_squared[:k].sum() / total_energy
                    layer_info[f"top{k}_energy"] = energy_k.item()

            results.append(layer_info)

    return pd.DataFrame(results)


if __name__ == "__main__":
    checkpoint_path = "skysense_checkpoint.pth"
    model = torch.load(checkpoint_path, map_location="cpu")

    df = compute_weight_spectrum(model)
    df.to_csv("analysis/weight_spectra_analysis.csv", index=False)

    print(df.head())