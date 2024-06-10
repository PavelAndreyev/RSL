_base_ = [
    'training_pipeline_tsm.py'
]

preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

checkpoint = ('https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth')
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MobileOneTSM',
        arch='s0',
        shift_div=8,
        num_segments=8,
        is_shift=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    cls_head=dict(
        type='TSMHead',
        num_segments=8,
        in_channels=1024,
        num_classes=34,
        spatial_type='avg',
        consensus=dict( type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
        average_clips='prob'),
    # model training and testing settings
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
    train_cfg=None,
    test_cfg=None)
