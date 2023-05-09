model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformerV2',
        arch='tiny',
        img_size=256,
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=7,
        in_channels=768,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=7,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=256,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=292,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackInputs')
]
train_dataloader = dict(
    batch_size=10,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='Data/ISIC_dataset',
        ann_file='train_ann.txt',
        data_prefix='Train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                scale=256,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies='timm_increasing',
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5,
                hparams=dict(pad_val=[104, 116, 124],
                             interpolation='bicubic')),
            dict(
                type='RandomErasing',
                erase_prob=0.25,
                mode='rand',
                min_area_ratio=0.02,
                max_area_ratio=0.3333333333333333,
                fill_color=[103.53, 116.28, 123.675],
                fill_std=[57.375, 57.12, 58.395]),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=10,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='Data/ISIC_dataset',
        ann_file='valid_ann.txt',
        data_prefix='Valid',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=292,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=256),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_dataloader = dict(
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
    batch_size=10,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='Data/ISIC_dataset',
        ann_file='test_ann.txt',
        data_prefix='Test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=292,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=256),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, ))
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=5,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-05, by_epoch=True, begin=1)
]
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=1024)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = 'work_dirs\\swinv2_tiny_ISICdataset\\train\\epoch_5.pth'
resume = False
randomness = dict(seed=None, deterministic=False)
launcher = 'none'
work_dir = 'work_dirs/swinv2_tiny_ISICdataset/test'
