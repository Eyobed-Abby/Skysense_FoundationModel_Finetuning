import os
import argparse
import torch
import torch.nn as nn
import pandas as pd

from models.swin_transformer_v2 import SwinTransformerV2


# ---------------------------------------------------------
# 1. Load Pretrained Backbone Only
# ---------------------------------------------------------

def load_pretrained_backbone(ckpt_path: str) -> nn.Module:
    model = SwinTransformerV2(
        arch="huge",
        img_size=224,
        patch_size=4,
        in_channels=3,
        drop_path_rate=0.2,
        window_size=8,
        out_indices=(3,),
        pad_small_map=True,
        pretrained_window_sizes=[0, 0, 0, 0],
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)

    return model


# ---------------------------------------------------------
# 2. Export Modules + Parameters
# ---------------------------------------------------------

def export_structure(model: nn.Module, out_modules: str, out_params: str):

    # ---------- MODULES ----------
    module_rows = []

    for name, module in model.named_modules():
        weight_shape = None
        bias_shape = None

        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            weight_shape = tuple(module.weight.shape)

        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            bias_shape = tuple(module.bias.shape)

        row = {
            "module_name": name,
            "module_type": type(module).__name__,
            "has_weight": weight_shape is not None,
            "weight_shape": weight_shape,
            "has_bias": bias_shape is not None,
            "bias_shape": bias_shape,
            "is_linear": isinstance(module, nn.Linear),
            "is_conv": isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)),
            "is_norm": isinstance(
                module,
                (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                 nn.LayerNorm, nn.GroupNorm)
            ),
            "is_attention_like": "attn" in name.lower(),
            "is_ffn_like": "ffn" in name.lower() or "mlp" in name.lower(),
        }

        module_rows.append(row)

    pd.DataFrame(module_rows).to_csv(out_modules, index=False)
    print(f"Saved module info → {out_modules}")

    # ---------- PARAMETERS ----------
    param_rows = []

    for name, param in model.named_parameters():
        if "." in name:
            module_name, param_name = name.rsplit(".", 1)
        else:
            module_name, param_name = "", name

        param_rows.append({
            "full_param_name": name,
            "module_name": module_name,
            "param_name": param_name,
            "shape": tuple(param.shape),
            "numel": int(param.numel()),
            "requires_grad": bool(param.requires_grad),
        })

    pd.DataFrame(param_rows).to_csv(out_params, index=False)
    print(f"Saved parameter info → {out_params}")


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_modules", default="analysis/backbone_modules.csv")
    parser.add_argument("--out_params", default="analysis/backbone_params.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_modules), exist_ok=True)

    print("Loading pretrained backbone only...")
    model = load_pretrained_backbone(args.ckpt)
    model.eval()

    export_structure(model, args.out_modules, args.out_params)


if __name__ == "__main__":
    main()