_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/isaid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
enable_tf32 = True
checkpoint_file = 'pretrain/skysense_model_backbone_hr.pth'
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(backbone=dict(
    _delete_=True,
    type='mmcls.SwinTransformerV2',
    arch='huge',
    img_size=224,
    patch_size=4,
    in_channels=3,
    window_size=8,
    drop_rate=0.,
    drop_path_rate=0.2,
    out_indices=(0, 1, 2, 3),
    use_abs_pos_embed=False,
    interpolate_mode='bicubic',
    with_cp=False,
    frozen_stages=-1,
    norm_eval=False,
    pad_small_map=False,
    norm_cfg=dict(type='LN'),
    stage_cfgs=dict(downsample_cfg=dict(is_post_norm=True)),
    patch_cfg=dict(),
    pretrained_window_sizes=[0, 0, 0, 0],
    init_cfg=dict(type='Pretrained',
                  checkpoint=checkpoint_file,
                  prefix='backbone',
                  map_location='cpu')),
             decode_head=dict(in_channels=[352, 704, 1408, 2816],
                              num_classes=16),
             auxiliary_head=dict(in_channels=1408, num_classes=16))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(_delete_=True,
                 type='AdamW',
                 lr=0.00006,
                 betas=(0.9, 0.999),
                 weight_decay=0.01,
                 constructor='SwinV2LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24,
                                    layer_decay_rate=0.9,
                                    depths=(2, 2, 18, 2),
                                    custom_keys={
                                        'bias': dict(decay_multi=0.),
                                        'absolute_pos_embed':
                                        dict(decay_mult=0.),
                                        'norm': dict(decay_mult=0.)
                                    }))

lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0,
                 min_lr=0.0,
                 by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)