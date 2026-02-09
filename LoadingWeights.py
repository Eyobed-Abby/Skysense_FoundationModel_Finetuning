import torch
from models.swin_transformer_v2 import SwinTransformerV2

# === Step 1: Initialize model ===
model = SwinTransformerV2(
    arch='huge',
    img_size=224,
    patch_size=4,
    in_channels=3,
    drop_path_rate=0.2,
    window_size=8,
    out_indices=(3,),
    pad_small_map=False,
    pretrained_window_sizes=[0, 0, 0, 0]
)

# === Step 2: Load checkpoint ===
ckpt_path = "skysense_model_backbone_hr.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

# === Step 3: Load with diagnostics ===
load_result = model.load_state_dict(state_dict, strict=False)
missing_keys = load_result.missing_keys
unexpected_keys = load_result.unexpected_keys

# === Step 4: Print summary ===
print(f"\n✅ Checkpoint loaded")
print(f"❌ Missing keys: {len(missing_keys)}")
print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")

print("\n--- First 10 Missing Keys ---")
for key in missing_keys[:10]:
    print("•", key)

print("\n--- All Unexpected Keys ---")
for key in unexpected_keys:
    print("•", key)
