from skysense_lora_classifier import build_lora_classifier

ckpt_path = "skysense_model_backbone_hr.pth"
model = build_lora_classifier(ckpt_path)
