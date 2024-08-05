# For SemanticKitti we usually do 19-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'SynLiDARDataset'
# data_root = 'data/semantickitti/'
data_root = '/data/SynLiDAR/'
# data_root = '/root/autodl-tmp//data/semantickitti'
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = {
    0: 19,  # "unlabeled"
    1: 0,  # "car"
    2: 3,  # "pick-up"
    3: 3,  # "truck"
    4: 4,  # "bus"
    5: 1,  # "bicycle"
    6: 2,  # "motorcycle"
    7: 4,  # "other-vehicle"
    8: 8,  # "road"
    9: 10,  # "sidewalk"
    10: 9,  # "parking"
    11: 11,  # "other-ground"
    12: 5,  # "female"
    13: 5,  # "male"
    14: 5,  # "kid"
    15: 5,  # "crowd"
    16: 6,  # "bicyclist"
    17: 7,  # "motorcyclist"
    18: 12,  # "building"
    19: 19,  # "other-structure"
    20: 14,  # "vegetation"
    21: 15,  # "trunk"
    22: 16,  # "terrain"
    23: 18,  # "traffic-sign"
    24: 17,  # "pole"
    25: 19,  # "traffic-cone"
    26: 13,  # "fence"
    27: 19,  # "garbage-can"
    28: 19,  # "electric-box"
    29: 19,  # "table"
    30: 19,  # "chair"
    31: 19,  # "bench"
    32: 19,  # "other-object"
}
metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=259)
input_modality = dict(use_lidar=True, use_camera=False)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/semantickitti/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity_minmax=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity_minmax=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity_minmax=True,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity_minmax=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[pcd_rotate_range, pcd_rotate_range],
                            scale_ratio_range=[
                                pcd_scale_factor, pcd_scale_factor
                            ],
                            translation_std=[0, 0, 0])
                        for pcd_rotate_range in [-0.78539816, 0.0, 0.78539816]
                        for pcd_scale_factor in [0.95, 1.0, 1.05]
                    ], [dict(type='Pack3DDetInputs', keys=['points'])]])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='synlidar_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='synlidar_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        test_mode=True,
        backend_args=backend_args))

val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel')
