import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from models.swin_transformer_v2 import SwinTransformerV2

class SkySenseClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 45):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2816, num_classes)  # SwinV2-Huge final dim

    def forward(self, x):
        feats = self.backbone(x)  # shape: [B, L, C]
        pooled = feats.mean(dim=1)  # global average over tokens
        return self.classifier(pooled)


def load_skysense_backbone(ckpt_path: str) -> nn.Module:
    model = SwinTransformerV2(
        arch='huge',
        img_size=224,
        patch_size=4,
        in_channels=3,
        drop_path_rate=0.2,
        window_size=8,
        out_indices=(3,),
        pad_small_map=True,
        pretrained_window_sizes=[0, 0, 0, 0]
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model


def build_lora_classifier_qkv(ckpt_path: str, lora_r: int = 8, lora_alpha: int = 16) -> nn.Module:
    backbone = load_skysense_backbone(ckpt_path)
    base_model = SkySenseClassifier(backbone)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "ffn.layers.0.0"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    model = get_peft_model(base_model, lora_config)
    model.forward = base_model.forward
    model.print_trainable_parameters()
    return model
