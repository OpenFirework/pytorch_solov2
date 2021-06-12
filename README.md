# Pytorch solov2 工程

本份代码从solov2的官方作者的原版代码中抽出来一部分，便于理解；  
目前只支持resnet18、resnet34两种backbone的训练和测试.   

## 安装
python 3.6+     
pip install torch==1.5.1  torchvision==0.6.1   #更高版本的pytorch经过测试也是OK的         
~~pip install mmcv~~     #会执行本地编译cuda代码，可能会花费10分钟左右, 2021-04-28更新，不再依赖mmcv，不需要安装此选项了       
pip install pycocotools      
pip install numpy   
pip install scipy  

- **必须**  
**安装SOLOV2原版的focalloss**  
cd pytorch_solov2/      
python setup.py develop  

后续会尝试剔除MMCV的依赖      

2020-07-23更新    
mmcv-full中的focalloss的实现与SOLO原版中的实现有差别（背景类的处理标签不同），如果使用mmcv-full的focalloss多次训练后，虽然损失下降，但实际预测不准确；     
因此替换为原本focalloss实现,安装好pytorch,cuda等必须的环境之后，项目根目录下执行python setup.py develop即可编译原版的focalloss    
替换之后，重新训练，损失和预测都正常

2020-10-13更新          
完善评测代码，保存为实例分割后的图片     
      
![avatar](results/00106.jpg)     

![avatar](results/00113.jpg)  


增加视频测试功能  


## 训练

配置好config中的项目之后，直接 python train.py      

如果配置coco训练集，则   
在data下ln -s /path/coco2017 coco    
修改配置文件coco2017_dataset中的字段,目前的项目配置是按照coco数据集配置，目录如下： 


例如使用本仓库自带的casia-SPT_val进行训练则设置如下：

- config.py中建立数据集的设置  
```python 
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
```
- dataset选择casia_SPT_val,设置batchsize(batchsize=imgs_per_gpu*workers_per_gpu)  

```python
solov2_base_config = coco_base_config.copy({
    'name': 'solov2_base',
 
    'backbone': resnet18_backbone,

    # Dataset stuff
    'dataset': casia_SPT_val,
    'num_classes': len(coco2017_dataset.class_names) + 1,
    'imgs_per_gpu': 6,
    'workers_per_gpu': 2,
    'num_gpus': 1,

```
完整的设置项， 在data/config.py中设置
```python
# ----------------------- SOLO v2.0 CONFIGS ----------------------- #

solov2_base_config = coco_base_config.copy({
    'name': 'solov2_base',
 
    'backbone': resnet18_backbone,

    # Dataset stuff
    'dataset': casia_SPT_val,
    'num_classes': len(coco2017_dataset.class_names) + 1,

     #batchsize=imgs_per_gpu*workers_per_gpu
    'imgs_per_gpu': 4,
    'workers_per_gpu': 2,
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
```



- 执行训练

```Python
python train.py  
```

## 评测
安装好环境后
```Python 
python eval.py 
```
可以评测coco数据格式的mAP,需要在配置文件中填入标签文件，图片路径等  
后续加入无标签文件纯图片生成mask和coco格式的json文件的代码;
eval.py 代码内部最后一句：
```Python 
eval(valmodel_weight='pretrained/solov2_448_r18_epoch_36.pth',data_path="data/casia-SPT_val/val/JPEGImages", benchmark=False, test_mode="images")
```
默认读取基于resnet18训练的网络，权重文件在pretraind文件夹中
基于resnet34和resnet50的训练完成的权重文件后续会放上链接：




## 注意

1.该网络训练时会将resnset的第一个卷积层和第一个stage的BasicBlock卷积层freeze,并且训练时resnet中的bn层都设置为eval()模式   
2. focalloss依赖mmcv的实现（2020-07-23更新，focalloss已经不依赖mmcv)，需要安装mmcv-full,但是另外一些预处理部分也有依赖，不过这部分也容易剔除掉   
3. 网络的输入要求长和宽都为32的倍数，这和他划分的网格有关，其他尺寸可能会无法计算 
4. 网络部分整体较为简单，没有奇怪的操作和层，前处理需要短边resize到448（保持比例），normalize，另外一边pad到32的倍数，后处理有一个卷积操作（卷积核心是训练时学习得到，在gpu上运算耗时基本很少），matrix_nms耗时也不多，经过该部分处理后，网络的输出为： 
 
 ```
mask [nums, height, weight]    #mask数量，图片原始高和宽，mask内容为0，1的二进制数据，1的位置表示mask 
cls [nums]                     #每个mask对应的分类
scores [nums]                  #每个mask对应的得分 
 ```
 

## 参考
https://github.com/WXinlong/SOLO   
https://github.com/open-mmlab/mmdetection   
https://github.com/open-mmlab/mmcv  


 
