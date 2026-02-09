import torch

def count_total_parameters(state_dict):
    total_params = 0
    param_stats = []

    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            numel = param.numel()
            total_params += numel
            param_stats.append((name, tuple(param.shape), numel))

    print(f"\n🔢 Total Parameters: {total_params:,}\n")
    print(f"{'Layer Name':60} {'Shape':>20} {'# Params':>12}")
    print("-" * 100)
    for name, shape, numel in param_stats[:30]:  # first 30 layers
        print(f"{name:60} {str(shape):>20} {numel:12,}")
    if len(param_stats) > 30:
        print(f"\n... and {len(param_stats) - 30} more layers.")
    
    return total_params

# === Load your checkpoint ===
ckpt_path = r"skysense_model_backbone_hr.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Unwrap if wrapped
if "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

# === Count total parameters ===
total = count_total_parameters(state_dict)
