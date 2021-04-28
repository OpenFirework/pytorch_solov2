import os
import numpy as np
from math import sqrt
import torch
from .piplines import LoadImageFromFile, LoadAnnotations, Normalize, DefaultFormatBundle, Collect, TestCollect, Resize, Pad, RandomFlip, MultiScaleFlipAug, ImageToTensor


process_funcs_dict = {'LoadImageFromFile':  LoadImageFromFile,
                      'LoadAnnotations': LoadAnnotations,
                      'Normalize': Normalize,
                      'DefaultFormatBundle': DefaultFormatBundle,
                      'Collect': Collect,
                      'TestCollect': TestCollect,
                      'Resize': Resize,
                      'Pad': Pad,
                      'RandomFlip': RandomFlip,
                      'MultiScaleFlipAug': MultiScaleFlipAug,
                      'ImageToTensor': ImageToTensor}

COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

# These are in RGB and are for ImageNet
MEANS = (123.675, 116.28, 123.675)
STD = (58.395, 57.12, 58.395)

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

class Config(object):
    """
    After implement this class, you can call 'cfg.x' instead of 'cfg['x']' to get a certain parameter.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making the changes given by new_config_dict.
        """
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object. Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def __repr__(self):
        return self.name
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


dataset_base = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'train_images': './data/coco/images/',
    'train_info':   'path_to_annotation_file',

    # Validation images and annotations.
    'valid_images': './data/coco/images/',
    'valid_info':   'path_to_annotation_file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': COCO_CLASSES,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
})

coco2017_dataset = dataset_base.copy({
   'name': 'COCO 2017',

    'train_prefix': './data/coco/',
    'train_info': 'annotations/instances_train2017.json',
    'trainimg_prefix': 'train2017/',
    'train_images': './data/coco/',

    'valid_prefix': './data/coco/',
    'valid_info': 'annotations/instances_val2017.json',
    'validimg_prefix': 'val2017/',
    'valid_images': './data/coco/',

    'label_map': COCO_LABEL_MAP


})

casia_SPT_val = dataset_base.copy({
    'name': 'casia-SPT 2020',
    
    'train_prefix': './data/casia-SPT_val/val/',
    'train_info': 'val_annotation.json',
    'trainimg_prefix': '',
    'train_images': './data/casia-SPT_val/val/',

    
    'valid_prefix': './data/casia-SPT_val/val/',
    'valid_info': 'val_annotation.json',
    'validimg_prefix': '',
    'valid_images': './data/casia-SPT_val/val',

    'label_map': COCO_LABEL_MAP
})

# ----------------------- BACKBONES ----------------------- #

backbone_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/pretrained/weights',
    'type': None,
})

resnet18_backbone = backbone_base.copy({
    'name': 'resnet18',
    'path': './pretrained/resnet18_nofc.pth',
    'type': 'ResNetBackbone',
    'num_stages': 4,
    'frozen_stages': 1,
    'out_indices': (0, 1, 2, 3)
})

resnet34_backbone = backbone_base.copy({
    'name': 'resnet34',
    'path': './pretrained/resnet34_nofc.pth',
    'type': 'ResNetBackbone',
    'num_stages': 4,
    'frozen_stages': 1,
    'out_indices': (0, 1, 2, 3)
})


#fpn config
fpn_base = Config({
    'in_channels': [64, 128, 256, 512],
    'out_channels': 256,
    'start_level': 0,
    'num_outs': 5,
})


# ----------------------- CONFIG DEFAULTS ----------------------- #

coco_base_config = Config({
    'dataset': coco2017_dataset,
    'num_classes': 81, # This should include the background class

})




# ----------------------- SOLO v2.0 CONFIGS ----------------------- #

solov2_base_config = coco_base_config.copy({
    'name': 'solov2_base',
 
    'backbone': resnet18_backbone,

    # Dataset stuff
    'dataset': casia_SPT_val,
    'num_classes': len(coco2017_dataset.class_names) + 1,

    'imgs_per_gpu': 2,
    'workers_per_gpu': 1,
    'num_gpus': 1,

    'train_pipeline':  [
        dict(type='LoadImageFromFile'),                                #read img process 
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),     #load annotations 
        dict(type='Resize',                                             #多尺度训练，随即从后面的size选择一个尺寸
            img_scale=[(768, 512), (768, 480), (768, 448),
                    (768, 416), (768, 384), (768, 352)],
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),                    #随机反转,0.5的概率
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),    #normallize                 
        dict(type='Pad', size_divisor=32),                                #pad另一边的size为32的倍数，solov2对网络输入的尺寸有要求，图像的size需要为32的倍数
        dict(type='DefaultFormatBundle'),                                #将数据转换为tensor，为后续网络计算
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')),   
    ],

    'test_cfg': None,

    # learning policy
    'lr_config': dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.01, step=[27, 33]),
    'total_epoch': 36,

    # optimizer
    'optimizer': dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),  

    'optimizer_config': dict(grad_clip=dict(max_norm=35, norm_type=2)),   #梯度平衡策略

    'resume_from': None,    #从保存的权重文件中读取，如果为None则权重自己初始化
    
    'epoch_iters_start': 1,    #本次训练的开始迭代起始轮数

    'test_pipeline': [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(768, 448),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ],

    'test_cfg': dict(
                nms_pre=500,
                score_thr=0.1,
                mask_thr=0.5,
                update_thr=0.05,
                kernel='gaussian',  # gaussian/linear
                sigma=2.0,
                max_per_img=30)

})

cfg = solov2_base_config.copy()


def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]

def set_dataset(dataset_name:str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)
