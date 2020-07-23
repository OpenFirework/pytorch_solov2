# Pytorch solov2 工程



## 安装
python 3.6+     
pip install torch==1.5.1  torchvision==0.6.1  
pip install mmcv     #会执行本地编译cuda代码，可能会花费10分钟左右
pip install pycocotools      
pip install numpy   
pip install scipy    


2020-07-23更新    
mmcv-full中的focalloss的实现与SOLO原版中的实现有差别（背景类的处理标签不同），如果使用mmcv-full的focalloss多次训练后，虽然损失下降，但实际预测不准确；     
因此替换为原本focalloss实现,安装好pytorch,cuda等必须的环境之后，项目根目录下执行python setup develop即可编译原版的focalloss    
替换之后，重新训练，损失和预测都正常   


## 训练

配置好config中的项目之后，直接 python train.py      

如果配置coco训练集，则   
在data下ln -s /path/coco2017 coco    
修改配置文件coco2017_dataset中的字段,目前的项目配置是按照coco数据集配置，目录如下：      


## 评测
安装好环境后，python eval.py 可以评测coco数据格式的mAP,需要在配置文件中填入标签文件，图片路径等  
后续加入无标签文件纯图片生成mask和coco格式的json文件的代码   



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

## 参考
https://github.com/WXinlong/SOLO   
https://github.com/open-mmlab/mmdetection   
https://github.com/open-mmlab/mmcv  


 
