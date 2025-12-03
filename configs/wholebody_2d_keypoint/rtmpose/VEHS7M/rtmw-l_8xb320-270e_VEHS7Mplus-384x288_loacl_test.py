_base_ = ['../../../_base_/default_runtime.py']

# common setting
num_keypoints = 37
sigmas = [0.025]*num_keypoints
# joint_weights = [
#     1.0,  # PELVIS
#     1.5,  # RWRIST
#     1.5,  # LWRIST
#     1.0,  # RHIP
#     1.0,  # LHIP
#     1.0,  # RKNEE
#     1.0,  # LKNEE
#     1.0,  # RANKLE
#     1.0,  # LANKLE
#     1.0,  # RFOOT
#     1.0,  # LFOOT
#     1.5,  # RHAND
#     1.5,  # LHAND
#     1.0,  # RELBOW
#     1.0,  # LELBOW
#     1.0,  # RSHOULDER
#     1.0,  # LSHOULDER
#     1.0,  # HEAD
#     1.0,  # THORAX
#     1.0,  # HDTP
#     1.0,  # REAR
#     1.0,  # LEAR
#     1.0,  # C7
#     1.0,  # C7_d
#     1.0,  # SS
#     1.0,  # RAP_b
#     1.0,  # RAP_f
#     1.0,  # LAP_b
#     1.0,  # LAP_f
#     1.0,  # RLE
#     1.0,  # RME
#     1.0,  # LLE
#     1.0,  # LME
#     1.5,  # RMCP2
#     1.5,  # RMCP5
#     1.5,  # LMCP2
#     1.5   # LMCP5
# ]
input_size = (288, 384)

# runtime
max_epochs = 10
stage2_num_epochs = 10
base_lr = 0.1 #5e-4
train_batch_size = 1
val_batch_size = 1

# new flag to freeze the neck
freeze_neck = True

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.1),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True,
        # custom_keys={
        #         # 'backbone': dict(lr_mult=0, decay_mult=0),
        #         'neck': dict(lr_mult=0, decay_mult=0)
        # }
        ))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=2560)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    decode_visibility=True)

find_unused_parameters = True
# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        channel_attention=True,
        frozen_stages=4,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'
            # checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth'  # noqa from https://github.com/open-mmlab/mmpose/tree/main/configs/wholebody_2d_keypoint/dwpose - DWPose-l	DW x-l	DW l-l	256x192 wb
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='neck.',
            checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'
        )),
    head=dict(
        type='RTMWHead',
        in_channels=1024,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=tuple([s // 32 for s in input_size]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=1.,
            label_softmax=True,
            label_beta=10.,
            # mask=list(range(23, 91)),  # for full body face keypoints in cocktail14 merge training
            # mask_weight=0.5,
        ),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'VEHS7M37kptsDataset'
data_mode = 'topdown'
# data_root = '/media/leyang/My Book/VEHS/'  # Linux path
data_root = '/home/leyang/Documents/mmpose/data/'
# data_root = '/nfs/turbo/coe-shdpm/leyang/'  # Slurm path

VEHS7M_train_ann_file = 'VEHS-7M/annotations/2D/VEHS_6DCOCO_downsample20_keep1_small_train.json'
VEHS7M_val_ann_file = 'VEHS-7M/annotations/2D/VEHS_6DCOCO_downsample20_keep1_small_validate.json'
# VEHS7M_train_ann_file = 'VEHS-7M/annotations/2D/VEHS_6DCOCO_downsample20_keep1_train.json'
# VEHS7M_val_ann_file = 'VEHS-7M/annotations/2D/VEHS_6DCOCO_downsample20_keep1_validate.json'

VEHS7M_metainfo = 'configs/_base_/datasets/VEHS7M_37kpts.py'
backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.55, 1.45], rotate_factor=85),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.675, 1.375],
        rotate_factor=75),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
        ]),
    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs')
]

# mapping
# (AIC-kpts, COCO133-kpts)
aic_coco133 = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12),
               (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)]

crowdpose_coco133 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                     (7, 12), (8, 13), (9, 14), (10, 15), (11, 16)]

mpii_coco133 = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
]

jhmdb_coco133 = [
    (3, 6),
    (4, 5),
    (5, 12),
    (6, 11),
    (7, 8),
    (8, 7),
    (9, 14),
    (10, 13),
    (11, 10),
    (12, 9),
    (13, 16),
    (14, 15),
]

halpe_coco133 = [(i, i)
                 for i in range(17)] + [(20, 17), (21, 20), (22, 18), (23, 21),
                                        (24, 19),
                                        (25, 22)] + [(i, i - 3)
                                                     for i in range(26, 136)]

posetrack_coco133 = [
    (0, 0),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
]

humanart_coco133 = [(i, i) for i in range(17)] + [(17, 99), (18, 120),
                                                  (19, 17), (20, 20)]
# convert
coco133_VEHS7M = [(11-1, 1),
                    (10-1, 2),
                    (13-1, 3),
                    (12-1, 4),
                    (15-1, 5),
                    (14-1, 6),
                    (17-1, 7),
                    (16-1, 8),
                    (21-1, 9),
                    (18-1, 10),
                    (122-1, 11),
                    (101-1, 12),
                    (9-1, 13),
                    (8-1, 14),
                    (7-1, 15),
                    (6-1, 16),
                    (51-1, 17),
                    (5-1, 20),
                    (4-1, 21),
                    (118-1, 33),
                    (130-1, 34),
                    (97-1, 35),
                    (109-1, 36)]
# ref: configs/_base_/datasets/VEHS7M_37kpts.py

aic_VEHS7M = [(0, 15), (1, 13), (2, 1), (3, 16), (4, 14), (5, 2), (6, 3), (7, 5), (8, 7), (9, 4), (10, 6), (11, 8)]

mpii_trb_VEHS7M = [(0, 16), (1, 15),
                   (2, 14), (3, 13),
                   (4, 2), (5, 1),
                   (6, 4), (7, 3),
                   (8, 6), (9, 5),
                   (10, 8), (11, 7),
                   (12, 19),
                   (13, 22)]

                   # (18, 30), (19, 29),  # elbow keypoints
                   # (20, 32),(21, 31)]

# [(0, 7), (1, 5), (2, 3), (4, 2), (5, 4), (10, 1), (11, 13), (12, 15), (13, 19), (14, 14), (15, 12)]

# convert others by other-coco133-VEHS7M?, wont work for some with coco133 missing points

# train datasets
dataset_VHES7M = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file=VEHS7M_train_ann_file,
    data_prefix=dict(img='VEHS-7M/img/5fps/train/'),
    pipeline=[],
)

dataset_coco = dict(
    type='CocoWholeBodyDataset',
    data_root=data_root+"Datasets/HumanPose2D/",
    data_mode=data_mode,
    ann_file='OpenDataLab___COCO-WholeBody/raw/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='OpenDataLab___COCO_2017/raw/Images/train2017'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=coco133_VEHS7M)
    ],
)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root+"Datasets/HumanPose2D/OpenDataLab___AI_Challenger/",
    data_mode=data_mode,
    ann_file='annotations/aic_train.json',
    data_prefix=dict(img='raw/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=aic_VEHS7M)
    ],
)


dataset_mpiitrb = dict(
    type='MpiiTrbDataset',
    data_root=data_root+"Datasets/HumanPose2D/OpenDataLab___MPII_Human_Pose/",
    data_mode=data_mode,
    ann_file='annotations/mpii_trb_train.json',
    data_prefix=dict(img='raw/images'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=mpii_trb_VEHS7M)
    ],
)

dataset_wb = dict(
    type='CombinedDataset',
    metainfo=dict(from_file=VEHS7M_metainfo),
    datasets=[dataset_VHES7M], #, dataset_aic, dataset_mpiitrb, dataset_coco],
    pipeline=[],
    test_mode=False,
)

train_datasets = [dataset_wb,]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file=VEHS7M_metainfo),
        datasets=train_datasets,
        pipeline=train_pipeline,
        test_mode=False,
    ))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file= VEHS7M_val_ann_file,
        data_prefix=dict(img='VEHS-7M/img/5fps/validate/'),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=10, interval=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + VEHS7M_val_ann_file,
    use_area=False)
test_evaluator = val_evaluator
