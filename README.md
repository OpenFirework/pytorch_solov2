# Pytorch solov2 工程

## 目录
```

|-- README.md
|-- data
|   |-- casia-SPT_val                                         #低功耗竞赛评测集200张     
|   |-- coco -> /media/data/data03/liujiangkuan/coco2017/     #coco数据集软连接          
|   |-- coco.py                                               #coco类型数据集解析功能代码         
|   |-- compose.py                                            #一些辅助功能函数      
|   |-- config.py                                             #配置文件          
|   |-- group_sampler.py                                      #数据集loader功能函数      
|   |-- loader.py                                             #数据集loader功能函数         
|   `-- piplines.py                                           #预处理相关功能函数，目前有些功能依赖mmcv，后续可剃出        
|-- eval.py                                                   #推理代码，待测试和补充，目前为空        
|-- mmdet2pythorch.py                                         #mmdet的原始权重转换为本项目权重的代码    
|-- modules         
|   |-- backbone.py
|   |-- focal_loss.py                                         #focaloss函数，依赖mmcv的实现
|   |-- mask_feat_head.py
|   |-- misc.py                                               #一些功能函数，包括matrix_nms，solov2中使用的nms方法  
|   |-- nninit.py                                             #一些初始化权重的方法     
|   |-- solov2.py                                             #整个solov2的网络   
|   `-- solov2_head.py                                        #solov2的预测分类和mask的分支的网络   
|-- pretrained                                                #预训练的权重   
|   |-- resnet18_nofc.pth                                     #torchvision的resnet18,34的权重   
|   |-- resnet34_nofc.pth                
|   |-- solov2_448_r18_epoch_36.pth                            #由solov2 mmdet原工程的权重文件，转换为而来   
|   `-- solov2_448_r34_epoch_36.pth                            #由solov2 mmdet原工程的权重文件，转换为而来    
|-- tools       
|-- train.py                                                    #训练脚本        
`-- weights         
```

## 安装
python 3.6+     
pip install torch==1.5.1  torchvision==0.6.1  
pip install mmcv-full     #会执行本地编译cuda代码，可能会花费10分钟左右
pip install pycocotools      
pip install numpy   
pip install scipy    
pip install Cython               #可能编译mmcv需要  

2020-07-23更新    
mmcv-full中的focalloss的实现与SOLO原版中的实现有差别（背景类的处理标签不同），如果使用mmcv-full的focalloss多次训练后，虽然损失下降，但实际预测不准确；     
因此替换为原本focalloss实现,安装好pytorch,cuda等必须的环境之后，项目根目录下执行python setup develop即可编译原版的focalloss    
替换之后，重新训练，损失和预测都正常   


## 训练

配置好config中的项目之后，直接 python train.py      

如果配置coco训练集，则   
在data下ln -s /media/data/data02/liujiangkuan/coco2017 coco    
修改配置文件coco2017_dataset中的字段,目前的项目配置是按照coco数据集配置，目录如下：    
测试项目目录位于117服务器， /home/rd5/liujiangkuan/project/new_solov2   


## 评测
安装好环境后，python eval.py 可以评测coco数据格式的mAP,需要在配置文件中填入标签文件，图片路径等  
后续加入无标签文件纯图片生成mask和coco格式的json文件的代码   


## 结果
resent18在竞赛200张评测集上的得分，在1050ti上的耗时 58.5 ms（推理+nms),对比Yolact resnet50的花费时间为70ms左右，而去年提交的tensorflow版本的yolact的耗时是在100ms以上，转换过程不完美，精度也有有一定的损失      
目前由二进制bitmask转换为cocoeval要求的格式的编码过程耗时较长，该部分可以借鉴yolact的解码部分进行优化  
作为对比 (去年的提交的程序在200张只有28map（resnet50)),在竞赛800张测试集合的分数只有23多
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.558
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.329
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.208
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.343
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.407
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.426
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.516
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.527
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.373
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.538
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.545
```
resent34在竞赛200张评测集上的得分,在1050ti上的耗时 68.6 ms（推理+nms)   
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.573
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.251
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.360
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.465
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.536
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.549
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.551
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.566
```


## 注意

1.该网络训练时会将resnset的第一个卷积层和第一个stage的BasicBlock卷积层freeze,并且训练时resnet中的bn层都设置为eval()模式   
2. focalloss依赖mmcv的实现，需要安装mmcv-full,另外一些预处理部分也有依赖，这部分容易剔除  
3. 网络的输入要求长和宽都为32的倍数，这和他划分的网格有关，其他尺寸可能会无法计算 
4. 网络部分整体较为简单，没有奇怪的操作和层，前处理需要短边resize到448（保持比例），normalize，另外一边pad到32的倍数，后处理有一个卷积操作（卷积核心是训练时学习得到，在gpu上运算耗时基本很少），matrix_nms耗时也不多，经过该部分处理后，网络的输出为： 
 
 ```
mask [nums, height, weight]    #mask数量，图片原始高和宽，mask内容为0，1的二进制数据，1的位置表示mask 
cls [nums]                     #每个mask对应的分类
scores [nums]                  #每个mask对应的得分 
 ```
 
 
5. 转onnx时，torch.linspace, group normal层好像不支持变成了ATen，后续待解决  
 
