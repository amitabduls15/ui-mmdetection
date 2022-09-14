_base_ = ['./faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py']

# load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
auto_resume = True
runner = dict(type='EpochBasedRunner', max_epochs=20)
workflow = [('train', 1), ('val', 1)]
# workflow = [('train', 1)]
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
evaluation = dict(interval=1, metric='bbox')
# 1. dataset settings

dataset_type = 'CocoDataset'
classes = ('HUMAN', 'OTH', 'ALB', 'YFT', 'BILL', 'LAG', 'SKJ', 'NoF', 'DOL',
           'BET', 'TUNA', 'PLS', 'SHARK', 'OIL', 'WATER')
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='data/coco-fishnet/train/labels.json',
        img_prefix='data/coco-fishnet/train/data'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='data/coco-fishnet/val/labels.json',
        img_prefix='data/coco-fishnet/val/data'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='data/coco-fishnet/test/labels.json',
        img_prefix='data/coco-fishnet/test/data'))

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=15)))

# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))  # handle exploding of gradient

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])