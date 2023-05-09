#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

_base_ = [
    '../_base_/models/swin_transformer_v2/tiny_256.py',
    '../_base_/datasets/imagenet_bs64_swin_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
model = dict(
    head=dict(
        num_classes=7,
    ),
)
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=7,
)
train_dataloader = dict(
    batch_size=10,
    dataset=dict(
        type=dataset_type,
        data_root='Data/ISIC_dataset',
        ann_file='train_ann.txt',
        data_prefix='Train',
    )
)
val_dataloader = dict(
    batch_size=10,
    dataset=dict(
        type=dataset_type,
        data_root='Data/ISIC_dataset',
        ann_file='valid_ann.txt',
        data_prefix='Valid',
    )
)
test_dataloader = dict(
    batch_size=10,
    dataset=dict(
        type=dataset_type,
        data_root='Data/ISIC_dataset',
        ann_file='test_ann.txt',
        data_prefix='Test',
    )
)
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator = val_evaluator
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=1)
]


train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
load_from = 'checkpoint\swinv2-tiny-w8_3rdparty_in1k-256px_20220803-e318968f.pth'
