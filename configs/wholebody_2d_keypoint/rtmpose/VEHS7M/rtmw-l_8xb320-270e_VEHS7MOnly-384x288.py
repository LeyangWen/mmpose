_base_ = ['../../../_base_/default_runtime.py']

# common setting
num_keypoints = 37
sigmas = [0.025]*num_keypoints
joint_weights = [1.]*num_keypoints
input_size = (288, 384)

# runtime
max_epochs = 270
stage2_num_epochs = 10
base_lr = 5e-4
train_batch_size = 320
val_batch_size = 32

# train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

max_epochs = 270
stage2_num_epochs = 30
train_cfg = dict(max_epochs=max_epochs, val_interval=20)


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.1),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

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
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth'  # noqa
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
        act_cfg=dict(type='SiLU', inplace=True)),
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
# data_root = '/media/leyang/My Book/VEHS/VEHS-7M/'  # Linux path
data_root = '/nfs/turbo/coe-shdpm/leyang/VEHS-7M/'  # Slurm path

# VEHS7M_train_ann_file = 'annotations/2D/VEHS_6DCOCO_downsample20_keep1_small_train.json'
# VEHS7M_val_ann_file = 'annotations/2D/VEHS_6DCOCO_downsample20_keep1_small_validate.json'
VEHS7M_train_ann_file = 'annotations/2D/VEHS_6DCOCO_downsample20_keep1_train.json'
VEHS7M_val_ann_file = 'annotations/2D/VEHS_6DCOCO_downsample20_keep1_validate.json'

VEHS7M_metainfo = 'configs/_base_/datasets/VEHS7M-37kpts.py'
backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
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
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
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


# train datasets
dataset_VHES7M = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file=VEHS7M_train_ann_file,
    data_prefix=dict(img='img/5fps/train/'),
    pipeline=[],
)


dataset_wb = dict(
    type='CombinedDataset',
    metainfo=dict(from_file=VEHS7M_metainfo),
    datasets=[dataset_VHES7M],
    pipeline=[],
    test_mode=False,
)

train_datasets = [dataset_wb,]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=2,
    pin_memory=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=VEHS7M_train_ann_file,
        data_prefix=dict(img='img/5fps/train/'),
        pipeline=train_pipeline,
        metainfo=dict(from_file=VEHS7M_metainfo),

        # type='CombinedDataset',
        # metainfo=dict(from_file=VEHS7M_metainfo),
        # datasets=train_datasets,
        # pipeline=train_pipeline,
        # test_mode=False,
    ))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file= VEHS7M_val_ann_file,
        data_prefix=dict(img='img/5fps/validate/'),
        pipeline=val_pipeline,
        # bbox_file='data/coco/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        test_mode=True))

test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))

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
