from data.config import cfg, process_funcs_dict
from data.coco import CocoDataset
from data.loader import build_dataloader
from modules.solov2 import SOLOV2
import torch.optim as optim
import time
import argparse
import torch
from torch.nn.utils import clip_grad
import pycocotools.mask as mask_util
import numpy as np
import mmcv
from data.coco_utils import  coco_eval, results2json, results2json_segm


def get_masks(result, num_classes=80):
    for cur_result in result:
        masks = [[] for _ in range(num_classes)]
        if cur_result is None:
            return masks
        seg_pred = cur_result[0].cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1].cpu().numpy().astype(np.int)
        cate_score = cur_result[2].cpu().numpy().astype(np.float)
        num_ins = seg_pred.shape[0]
        for idx in range(num_ins):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks[cate_label[idx]].append(rst)

        return masks


#set requires_grad False
def gradinator(x):
    x.requires_grad = False
    return x


def build_process_pipeline(pipeline_confg):
    assert isinstance(pipeline_confg, list)
    process_pipelines = []
    for pipconfig in pipeline_confg:
        assert isinstance(pipconfig, dict) and 'type' in pipconfig
        args = pipconfig.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            process_pipelines.append(process_funcs_dict[obj_type](**args))
            
    return process_pipelines


def eval(valmodel_weight):
    test_pipelines = []
    loadimg = process_funcs_dict['LoadImageFromFile']()
    test_pipelines.append(loadimg)

    transforms=[ dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ]

    transforms_piplines = build_process_pipeline(transforms)
    Multest = process_funcs_dict['MultiScaleFlipAug'](transforms = transforms_piplines, img_scale = (768, 448), flip=False)
    test_pipelines.append(Multest)

     # #build datashet
    casiadata = CocoDataset(ann_file=cfg.dataset.valid_info,
                            pipeline = test_pipelines,
                            img_prefix = cfg.dataset.validimg_prefix,
                            data_root=cfg.dataset.valid_prefix,
                            test_mode=True)

    torchdata_loader = build_dataloader(casiadata, imgs_per_gpu=1, workers_per_gpu=cfg.workers_per_gpu, num_gpus=1, shuffle=False)

    model = SOLOV2(cfg, pretrained = valmodel_weight, mode='test')
    model = model.cuda()

    results = []
    for k, enumdata in enumerate(torchdata_loader):

        imgs = enumdata['img']
        inputs = []
        for img in imgs:
            inputs.append(img.cuda())
        
        img_infos = []
        img_metas = enumdata['img_metas']   #图片的一些原始信息
        for img_meta in img_metas:
            img_infos.append(img_meta.data[0])
        
        with torch.no_grad():      
            seg_result = model.forward(img=inputs, img_meta=img_infos, return_loss=False)
        #mask [nums, height, weight] 
        #cls [nums]
        #scores [nums]
        result = get_masks(seg_result, num_classes=80)

        results.append(result)

    eval_types = ['segm']
    mmcv.dump(results, "reseg_results.pkl")
    if not isinstance(results[0], dict):
        result_files = results2json_segm(casiadata, results, "reseg_results.pkl")
        coco_eval(result_files, eval_types, casiadata.coco)
    else:
        for name in results[0]:
            print('\nEvaluating {}'.format(name))
            outputs_ = [out[name] for out in results]
            result_file = "reseg_results.json" + '.{}'.format(name)
            result_files = results2json(casiadata, outputs_, result_file)
            coco_eval(result_files, eval_types, casiadata.coco)

eval(valmodel_weight='pretrained/solov2_448_r18_epoch_36.pth')