_base_ = [
    '/home/ppds/PycharmProjects/mmaction2/configs/_base_/models/tsm_mobilenetv2.py',
    '/home/ppds/PycharmProjects/mmaction2/configs/_base_/schedules/sgd_50e.py',
    '/home/ppds/PycharmProjects/mmaction2/configs/_base_/default_runtime.py'
]


# Путь к видео
data_root = '/home/ppds/PycharmProjects/gluh'
video_prefix = ''

dataset_type = 'VideoDataset'
classes = 34   # Кол-во букв

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

data_preprocessor = dict(
    type='ActionDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    format_shape='NCHW',
    input_size=224)

model = dict(
    backbone=dict(pretrained=None),
    cls_head=dict(num_classes=34),
    test_cfg=dict(
        average_clips='prob'  # объединяет клипы по вероятностям, не по логитам
    )
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=f'{data_root}/train.txt',
        data_prefix=dict(video=video_prefix),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=f'{data_root}/val.txt',
        data_prefix=dict(video=video_prefix),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='AccMetric',
    collect_device='cpu'
)


test_evaluator = val_evaluator
test_pipeline = [
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]



default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

log_level = 'INFO'
load_from = None
work_dir = './work_dirs/tsm_bukva'
