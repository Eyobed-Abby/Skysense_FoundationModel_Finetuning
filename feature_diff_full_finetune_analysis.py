import os
import re
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.optim import AdamW

from models.swin_transformer_v2 import SwinTransformerV2
from resisc45_loader import get_resisc45_dataloaders


# ==========================================================
# 1. Reproducibility
# ==========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# 2. SkySense-style classifier (NO LoRA)
# ==========================================================

class SkySenseClassifier(nn.Module):
    """
    Plain classifier: SwinV2 backbone + linear head.
    No LoRA, no PEFT, just full model.
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 45):
        super().__init__()
        self.backbone = backbone
        # For SwinV2 huge used in your code, final dim is 2816
        self.classifier = nn.Linear(2816, num_classes)

    def forward(self, x):
        feats = self.backbone(x)

        # Some implementations return a tuple (features, ...)
        if isinstance(feats, tuple):
            feats = feats[0]

        # [B, C, H, W] -> GAP -> [B, C]
        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1)
            feats = feats.squeeze(-1).squeeze(-1)
        elif feats.ndim != 2:
            raise ValueError(f"Unexpected backbone output shape: {feats.shape}")

        logits = self.classifier(feats)
        return logits


def load_swin_backbone(ckpt_path: str) -> nn.Module:
    """
    Load SwinTransformerV2 'huge' backbone with pretrained weights.
    Mirrors how you're using it elsewhere, but WITHOUT LoRA.
    """
    backbone = SwinTransformerV2(
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
    backbone.load_state_dict(state_dict, strict=False)
    return backbone


def build_plain_classifier(ckpt_path: str, num_classes: int = 45) -> nn.Module:
    backbone = load_swin_backbone(ckpt_path)
    model = SkySenseClassifier(backbone, num_classes=num_classes)
    return model


# ==========================================================
# 3. Layer matching / hooks (post-activation features)
# ==========================================================

def match_layers(name: str) -> bool:
    """
    Decide which modules to hook based on their dotted name.
    We only hook backbone blocks, attn, qkv, and ffn.
    """
    patterns = [
        r"backbone\.stages\.\d+\.blocks\.\d+$",                      # whole block
        r"backbone\.stages\.\d+\.blocks\.\d+\.attn$",                # attention
        r"backbone\.stages\.\d+\.blocks\.\d+\.attn\.w_msa\.qkv$",    # qkv
        r"backbone\.stages\.\d+\.blocks\.\d+\.ffn$",                 # ffn
    ]
    return any(re.search(p, name) for p in patterns)


def register_hooks(model: nn.Module):
    """
    Register forward hooks on selected layers.

    We record the POST-ACTIVATION feature map for each module:
      - [B, C, H, W] -> global average pool -> [B, C]
      - [B, T, C]    -> mean over T        -> [B, C]

    Returns:
        activations: dict[layer_name] -> list of [B, C] tensors (CPU)
        handles: list of hook handles
    """
    activations = {}
    handles = []

    def hook_fn(name):
        def hook(module, input, output):
            out = output
            if isinstance(out, tuple):
                out = out[0]

            # [B, C, H, W] -> global avg pool -> [B, C]
            if out.ndim == 4:
                out = F.adaptive_avg_pool2d(out, 1)
                out = out.squeeze(-1).squeeze(-1)
            # [B, T, C] -> mean over T -> [B, C]
            elif out.ndim == 3:
                out = out.mean(dim=1)

            if out.ndim != 2:
                raise ValueError(
                    f"Hooked layer {name} produced tensor of shape {out.shape}, "
                    f"expected [B, C] after pooling/averaging."
                )

            activations.setdefault(name, []).append(out.detach().cpu())
        return hook

    for name, module in model.named_modules():
        if match_layers(name):
            handles.append(module.register_forward_hook(hook_fn(name)))

    return activations, handles


# ==========================================================
# 4. Utility: weight shape (optional)
# ==========================================================

def get_weight_shape(model: nn.Module, layer_name: str):
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            return tuple(module.weight.shape)
    return None


# ==========================================================
# 5. Spectral / intrinsic-dimension metrics
# ==========================================================

def compute_spectrum(D: torch.Tensor):
    """
    D: [num_samples, feat_dim] feature difference matrix for a layer.

    Returns dict with:
        - feat_dim, num_eigs, full_rank
        - stable_rank, participation_ratio, effective_rank
        - dim_30, dim_50, dim_70, dim_90, dim_95
        - top4/8/16/32/64_energy
        - top30/50/70/90_frac (fraction of eigenvalues needed to reach
          30%, 50%, 70%, 90% of the total variance of D).
    """
    # Center along samples (remove mean feature)
    D = D - D.mean(dim=0, keepdim=True)

    # SVD: works for tall or wide matrices
    try:
        U, S, V = torch.linalg.svd(D, full_matrices=False)
    except torch.linalg.LinAlgError:
        # In rare numerical issues, fall back to CPU double
        D_cpu = D.to(dtype=torch.float64, device="cpu")
        U, S, V = torch.linalg.svd(D_cpu, full_matrices=False)

    eigvals = (S ** 2).to(torch.float64)

    num_eigs = S.numel()
    feat_dim = D.shape[1]

    # Handle degenerate case: all zeros
    total = eigvals.sum()
    if total <= 0:
        return {
            "feat_dim": int(feat_dim),
            "num_eigs": int(num_eigs),
            "full_rank": 0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "effective_rank": 0.0,
            "dim_30": 0,
            "dim_50": 0,
            "dim_70": 0,
            "dim_90": 0,
            "dim_95": 0,
            "top4_energy": float("nan"),
            "top8_energy": float("nan"),
            "top16_energy": float("nan"),
            "top32_energy": float("nan"),
            "top64_energy": float("nan"),
            "top30_frac": 0.0,
            "top50_frac": 0.0,
            "top70_frac": 0.0,
            "top90_frac": 0.0,
        }

    # Normalized spectrum p_i and cumulative variance
    ratios = eigvals / (total + 1e-12)
    cumulative = torch.cumsum(ratios, dim=0)

    def dim_for(thr: float) -> int:
        idx = (cumulative >= thr).nonzero(as_tuple=True)[0]
        return int(idx[0]) + 1 if idx.numel() > 0 else num_eigs

    def topk_energy(k: int) -> float:
        if num_eigs == 0 or k <= 0:
            return float("nan")
        k = min(k, num_eigs)
        return float(cumulative[k - 1].item())

    # Dimensionality thresholds
    dim_30 = dim_for(0.30)
    dim_50 = dim_for(0.50)
    dim_70 = dim_for(0.70)
    dim_90 = dim_for(0.90)
    dim_95 = dim_for(0.95)

    # Stable / participation / effective rank
    stable_rank = float((eigvals.sum() / (eigvals.max() + 1e-12)).item())
    participation_ratio = float((eigvals.sum() ** 2 / (eigvals ** 2).sum()).item())

    p = eigvals / (eigvals.sum() + 1e-12)
    entropy_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())

    # Fractions of eigenvalues needed (your requested metrics)
    top30_frac = dim_30 / float(num_eigs)
    top50_frac = dim_50 / float(num_eigs)
    top70_frac = dim_70 / float(num_eigs)
    top90_frac = dim_90 / float(num_eigs)

    return {
        "feat_dim": int(feat_dim),
        "num_eigs": int(num_eigs),
        "full_rank": int((S > 1e-6).sum().item()),
        "stable_rank": stable_rank,
        "participation_ratio": participation_ratio,
        "effective_rank": entropy_rank,
        "dim_30": dim_30,
        "dim_50": dim_50,
        "dim_70": dim_70,
        "dim_90": dim_90,
        "dim_95": dim_95,
        "top4_energy": topk_energy(4),
        "top8_energy": topk_energy(8),
        "top16_energy": topk_energy(16),
        "top32_energy": topk_energy(32),
        "top64_energy": topk_energy(64),
        "top30_frac": top30_frac,
        "top50_frac": top50_frac,
        "top70_frac": top70_frac,
        "top90_frac": top90_frac,
    }


# ==========================================================
# 6. Main: train tuned model briefly, then analyze feature diff
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True,
                        help="Pretrained backbone checkpoint (base model).")
    parser.add_argument("--num_classes", type=int, default=45)
    parser.add_argument("--train_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=1,
                        help="Number of epochs of full finetuning to run.")
    parser.add_argument("--max_train_batches", type=int, default=-1,
                        help="Optional cap on train batches per epoch (-1 = all).")
    parser.add_argument("--analysis_batches", type=int, default=20,
                        help="How many batches to use for feature analysis.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="analysis/one_epoch_full_ft_feature_diff.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    # Ensure output directory exists
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------
    # Build base & tuned models (no LoRA)
    # ------------------------------------------------------
    print("Building base (pretrained) and tuned (trainable) models...")
    base_model = build_plain_classifier(args.base_ckpt, num_classes=args.num_classes).to(device)
    tuned_model = build_plain_classifier(args.base_ckpt, num_classes=args.num_classes).to(device)

    # base_model is never updated
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    tuned_model.train()
    optimizer = AdamW(tuned_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------
    # Data loader for training
    # ------------------------------------------------------
    print("Preparing data loaders...")
    train_loader, _ = get_resisc45_dataloaders(
        train_split=args.train_split,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------
    # Brief full finetuning (few steps / one epoch)
    # ------------------------------------------------------
    print(f"Starting full finetuning for {args.train_epochs} epoch(s)...")
    for epoch in range(args.train_epochs):
        tuned_model.train()
        total_loss = 0.0
        num_batches = 0

        for i, (x, y) in enumerate(train_loader):
            if args.max_train_batches > 0 and i >= args.max_train_batches:
                break

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = tuned_model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
        else:
            avg_loss = float("nan")
        print(f"Epoch {epoch+1}/{args.train_epochs} - avg train loss: {avg_loss:.4f}")

    tuned_model.eval()
    print("Finished brief full finetuning. Starting feature-diff analysis...")

    # ------------------------------------------------------
    # Rebuild data loader for analysis (fresh iterator)
    # ------------------------------------------------------
    train_loader, _ = get_resisc45_dataloaders(
        train_split=args.train_split,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------
    # Register hooks
    # ------------------------------------------------------
    base_acts, base_handles = register_hooks(base_model)
    tuned_acts, tuned_handles = register_hooks(tuned_model)

    # ------------------------------------------------------
    # Run a few batches through both models for analysis
    # ------------------------------------------------------
    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            if i >= args.analysis_batches:
                break
            x = x.to(device)
            base_model(x)
            tuned_model(x)

    # Remove hooks
    for h in base_handles + tuned_handles:
        h.remove()

    # ------------------------------------------------------
    # Per-layer analysis: base vs 1-epoch full FT tuned
    # ------------------------------------------------------
    rows = []

    for layer in sorted(base_acts.keys()):
        if layer not in tuned_acts:
            continue

        # Concatenate all batches for this layer
        F_pre = torch.cat(base_acts[layer], dim=0)   # [N, C] from pretrained
        F_post = torch.cat(tuned_acts[layer], dim=0) # [N, C] from briefly full-FT model

        # Sanity check: shapes must match
        if F_pre.shape != F_post.shape:
            raise RuntimeError(
                f"Shape mismatch for layer {layer}: "
                f"pre={F_pre.shape}, post={F_post.shape}"
            )

        # Feature difference (updated feature map)
        D = F_post - F_pre  # change in post-activation features

        # Norms of update
        fro = torch.norm(D)
        rel = fro / (torch.norm(F_pre) + 1e-8)

        # Spectral / intrinsic-dimension metrics
        spec = compute_spectrum(D.to(device))

        # Optional: weight shape from backbone
        w_shape = get_weight_shape(base_model, layer)

        row = {
            "layer": layer,
            "weight_shape": w_shape,
            "feature_shape": tuple(D.shape),
            "num_samples": int(D.shape[0]),
            "fro_change": float(fro.item()),
            "relative_change": float(rel.item()),
        }
        row.update(spec)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Saved feature-diff intrinsic-dimension CSV to:", args.out)


if __name__ == "__main__":
    main()