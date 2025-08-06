_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'pretrain/skysense_model_backbone_hr.pth'
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(type='FasterRCNN',
             backbone=dict(
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
                               checkpoint=pretrained,
                               prefix='backbone',
                               map_location='cpu')),
             neck=dict(in_channels=[352, 704, 1408, 2816]),
             roi_head=dict(bbox_head=dict(num_classes=20), ))

optimizer = dict(_delete_=True,
                 type='AdamW',
                 lr=0.0001,
                 betas=(0.9, 0.999),
                 weight_decay=0.05,
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
lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)
data = dict(samples_per_gpu=1)
