_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/default_runtime.py'
]

model = dict(
    type='MinkUNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='minkunet',
        batch_first=False,
        max_voxels=80000,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-100, -100, -20, 100, 100, 20],
            voxel_size=[0.05, 0.05, 0.05],
            max_voxels=(-1, -1))),
    backbone=dict(
        type='SPVCNNBackbone',
        in_channels=4,
        num_stages=4,
        base_channels=16,
        encoder_channels=[16, 32, 64, 128],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[128, 64, 48, 48],
        decoder_blocks=[2, 2, 2, 2],
        block_type='basic',
        sparseconv_backend='torchsparse',
        drop_ratio=0.3),
    decode_head=dict(
        type='MinkUNetHead',
        channels=48,
        num_classes=19,
        batch_first=False,
        dropout_ratio=0,
        loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True),
        ignore_index=19),
    train_cfg=dict(),
    test_cfg=dict())

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti'),
    dict(type='PointSegClassMapping'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(
    sampler=dict(seed=0), dataset=dict(pipeline=train_pipeline))

lr = 0.24
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='SGD', lr=lr, weight_decay=0.0001, momentum=0.9, nesterov=True))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.008, by_epoch=False, begin=0, end=125),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=15,
        by_epoch=True,
        eta_min=1e-5,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))
randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
env_cfg = dict(cudnn_benchmark=True)


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')