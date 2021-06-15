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
import cv2 as cv
from data.compose import Compose
from glob import glob
import pycocotools.mask as maskutil
import json
import os
from scipy import ndimage
from data.imgutils import rescale_size, imresize, imrescale, imflip, impad, impad_to_multiple


COCO_LABEL = [1,  2,  3,  4,  5,  6,  7,  8,
                   9, 10, 11, 13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23, 24, 25,
                  27, 28, 31, 32, 33, 34, 35, 36,
                  37, 38, 39, 40, 41, 42, 43, 44,
                  46, 47, 48, 49, 50, 51, 52, 53,
                  54, 55, 56, 57, 58, 59, 60, 61,
                  62, 63, 64, 65, 67, 70, 72, 73,
                  74, 75, 76, 77, 78, 79, 80, 81,
                  82, 84, 85, 86, 87, 88, 89, 90]

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

CLASS_NAMES=(COCO_CLASSES, COCO_LABEL)

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

def result2json(img_id, result):
    rel = []
    seg_pred = result[0][0].cpu().numpy().astype(np.uint8)
    cate_label = result[0][1].cpu().numpy().astype(np.int)
    cate_score = result[0][2].cpu().numpy().astype(np.float)
    num_ins = seg_pred.shape[0]
    for j in range(num_ins):
        realclass = COCO_LABEL[cate_label[j]]
        re = {}
        score = cate_score[j]
        re["image_id"] = img_id
        re["category_id"] = int(realclass)
        re["score"] = float(score)
        outmask = np.squeeze(seg_pred[j])
        outmask = outmask.astype(np.uint8)
        outmask=np.asfortranarray(outmask)
        rle = maskutil.encode(outmask)
        rle['counts'] = rle['counts'].decode('ascii')
        re["segmentation"] = rle
        rel.append(re)
    return rel

class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None 
        img = cv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

class LoadImageInfo(object):
    def __call__(self, frame):
        results={}
        results['filename'] = None 
        results['img'] = frame
        results['img_shape'] = frame.shape
        results['ori_shape'] = frame.shape
        return results


def show_result_ins(img,
                    result,
                    score_thr=0.3,
                    sort_by_density=False):
    if isinstance(img, str):
        img = cv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    #img_show = None
    for idx in range(num_mask):
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        #当前实例的类别
        cur_cate = cate_label[idx]
        realclass = COCO_LABEL[cur_cate]
        cur_score = cate_score[idx]

        name_idx = COCO_LABEL_MAP[realclass]
        label_text = COCO_CLASSES[name_idx-1]
        label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv.putText(img_show, label_text, vis_pos,
                        cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
 
    return img_show


def eval(valmodel_weight, data_path, benchmark, test_mode, save_imgs=False):
    
    test_pipeline = []
    transforms=[ dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='TestCollect', keys=['img']),
    ]
    transforms_piplines = build_process_pipeline(transforms)
    Multest = process_funcs_dict['MultiScaleFlipAug'](transforms = transforms_piplines, img_scale = (480, 448), flip=False)

    if test_mode == "video":
        test_pipeline.append(LoadImageInfo())
    elif test_mode == "images":
        test_pipeline.append(LoadImage())
    else:
        raise NotImplementedError("not support mode!")
    test_pipeline.append(Multest)
    test_pipeline = Compose(test_pipeline)

    model = SOLOV2(cfg, pretrained=valmodel_weight, mode='test')
    model = model.cuda()


    if test_mode == "video":
        vid = cv.VideoCapture(data_path)
        target_fps   = round(vid.get(cv.CAP_PROP_FPS))
        frame_width  = round(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = round(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        num_frames = round(vid.get(cv.CAP_PROP_FRAME_COUNT))
        
        for i in range(num_frames):
            if i%5 != 0:
                continue
            frame=vid.read()
            img=frame[1]
            data = test_pipeline(img)
            imgs = data['img']

            img = imgs[0].cuda().unsqueeze(0)
            img_info = data['img_metas']
            start = time.time()
            with torch.no_grad():
                seg_result = model.forward(img=[img], img_meta=[img_info], return_loss=False)
            
            img_show = show_result_ins(frame[1],seg_result)
            end = time.time()
            print("spend time: ",(end-start))
            cv.imshow("watch windows",img_show)
            cv.waitKey(1)

    if test_mode == "images":
        img_ids = []
        images = []
        use_json = False
        imgs_prefix = "tests"
        test_imgpath = data_path
        if use_json == False:
            test_imgpath = test_imgpath + '/*'
            images = glob(test_imgpath)
            for img in images:
                pathname,filename = os.path.split(img)
                prefix,suffix = os.path.splitext(filename)
                img_id = int(prefix)
                img_ids.append(str(img_id))  
        else:
            imgsinfo=json.load(open("data/casia-SPT_val/val/val_annotation.json",'r'))
            for i in range(len(imgsinfo['images'])):
                img_id = imgsinfo['images'][i]['id']
                img_path = imgs_prefix + '/'+ imgsinfo['images'][i]['file_name']
                img_ids.append(img_id)
                images.append(img_path)

        imgs_nums = len(images)
        results = []
        k = 0
        for imgpath in images:
            img_id = img_ids[k]
            data = dict(img=imgpath)
            data = test_pipeline(data)
            imgs = data['img']

            img = imgs[0].cuda().unsqueeze(0)
            img_info = data['img_metas']
            with torch.no_grad():
                seg_result = model.forward(img=[img], img_meta=[img_info], return_loss=False)
            img_show = show_result_ins(imgpath,seg_result)

            #cv.imshow("watch windows",img_show)
            #cv.waitKey(1)
            out_filepath = "results/" + os.path.basename(imgpath)

            k = k + 1
            if save_imgs:
                cv.imwrite(out_filepath, img_show)
            if benchmark == True:
                result = result2json(img_id, seg_result)
                results = results + result

        if benchmark == True:
            re_js = json.dumps(results)
            fjson = open("eval_masks.json","w")
            fjson.write(re_js)
            fjson.close()

eval(valmodel_weight='pretrained/solov2_448_r18_epoch_36.pth',data_path="data/casia-SPT_val/val/JPEGImages", benchmark=False, test_mode="images", save_imgs=False)
#eval(valmodel_weight='pretrained/solov2_448_r18_epoch_36.pth',data_path="cam0.avi", benchmark=False, test_mode="video")
