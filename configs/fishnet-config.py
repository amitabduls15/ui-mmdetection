_base_ = './faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

# 1. dataset settings

dataset_type = 'CocoDataset'
classes = ('HUMAN', 'OTH', 'ALB', 'YFT', 'BILL', 'LAG', 'SKJ', 'NoF', 'DOL',
'BET', 'TUNA', 'PLS', 'SHARK', 'OIL', 'WATER')
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
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
    pretrained=None,
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',

            num_classes=15,

           )))