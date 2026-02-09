import torch
from models.swin_transformer_v2 import SwinTransformerV2  # Make sure this file is in your PYTHONPATH or same dir

# === Step 1: Create SwinV2-Huge model ===
model = SwinTransformerV2(
    arch='huge',
    img_size=224,
    patch_size=4,
    in_channels=3,
    drop_path_rate=0.2,   # from SkySense paper
    window_size=8,
    out_indices=(3,),     # output from last stage
    pad_small_map=False,
    pretrained_window_sizes=[0, 0, 0, 0]  # disable interpolation
)

# === Step 2: Load Checkpoint ===
ckpt_path = r"skysense_model_backbone_hr.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")

# If it's wrapped
if "model" in state_dict:
    state_dict = state_dict["model"]

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("✅ Checkpoint loaded")
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
