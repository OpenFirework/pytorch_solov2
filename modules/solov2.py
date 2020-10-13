import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet34, resnet50
from .nninit import xavier_init, kaiming_init
from .solov2_head import SOLOv2Head
from .mask_feat_head import MaskFeatHead
import torch.distributed as dist
import torch.multiprocessing as m
from itertools import chain

from torch.nn.parallel import DataParallel



class FPN(nn.Module):
    
    def __init__(self, 
               in_channels,
               out_channels,
               num_outs,
               start_level=0,
               end_level=-1,
               add_extra_convs=False,
               extra_convs_on_inputs=True,
               relu_before_extra_convs=False,
               no_norm_on_lateral=False,
               upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels  #[64, 128, 256, 512] resnet18
        self.out_channels = out_channels  #256
        self.num_ins = len(in_channels)  #4 
        self.num_outs = num_outs        #5 
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        if end_level == -1:                  #default -1
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
  
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:   #default false
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
 


class SOLOV2(nn.Module):
    
    def __init__(self,
                 cfg=None,
                 pretrained=None,
                 mode='train'):
        super(SOLOV2, self).__init__()
        if cfg.backbone.name == 'resnet18':
            self.backbone = resnet18(pretrained=True, loadpath = cfg.backbone.path)
        elif cfg.backbone.name == 'resnet34':
            self.backbone = resnet34(pretrained=True, loadpath = cfg.backbone.path)
        else:
            raise NotImplementedError
        
        #this set only support resnet18 and resnet34 backbone, 可以根据solo中resent50的配置进行更改，使其支持resnset50的训练，下同
        self.fpn = FPN(in_channels=[64, 128, 256, 512],out_channels=256,start_level=0,num_outs=5,upsample_cfg=dict(mode='nearest'))

        #this set only support resnet18 and resnet34 backbone
        self.mask_feat_head = MaskFeatHead(in_channels=256,
                            out_channels=128,
                            start_level=0,
                            end_level=3,

                            num_classes=128)
        #this set only support resnet18 and resnet34 backbone
        self.bbox_head = SOLOv2Head(num_classes=81,
                            in_channels=256,
                            seg_feat_channels=256,
                            stacked_convs=2,
                            strides=[8, 8, 16, 32, 32],
                            scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                            num_grids=[40, 36, 24, 16, 12],
                            ins_out_channels=128
                        )
        
        self.mode = mode

        self.test_cfg = cfg.test_cfg

        if self.mode == 'train':
            self.backbone.train(mode=True)
        else:
            self.backbone.train(mode=True)
        
        if pretrained is None:
            self.init_weights() #if first train, use this initweight
        else:
            self.load_weights(pretrained)             #load weight from file
    
    def init_weights(self):
        #fpn
        if isinstance(self.fpn, nn.Sequential):
            for m in self.fpn:
                m.init_weights()
        else:
            self.fpn.init_weights()
        
        #mask feature mask
        if isinstance(self.mask_feat_head, nn.Sequential):
            for m in self.mask_feat_head:
                m.init_weights()
        else:
            self.mask_feat_head.init_weights()

        self.bbox_head.init_weights()
    
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
 
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.fpn(x)
        return x
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):


        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
        loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas)

        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

  
    # 短边resize到448，剩余的边pad到能被32整除
    '''
    img_metas context
    'filename': 'data/casia-SPT_val/val/JPEGImages/00238.jpg', 
    'ori_shape': (402, 600, 3), 'img_shape': (448, 669, 3), 
    'pad_shape': (448, 672, 3), 'scale_factor': 1.1144278606965174, 'flip': False, 
    'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 
    'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}

    '''

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)
    
    def simple_test(self, img, img_meta, rescale=False):
       
        #test_tensor = torch.ones(1,3,448,512).cuda()
        #x = self.extract_feat(test_tensor)
        x = self.extract_feat(img)

        outs = self.bbox_head(x,eval=True)
  
        mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])

        seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)

        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError


# not use 
'''
def get_dist_info():

    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def load_state_dict(module, state_dict, strict=False):
   
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(filename, map_location=None):
    
    if not osp.isfile(filename):
        raise IOError(f'{filename} is not a checkpoint file')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False):

    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    load_state_dict(model, state_dict, strict)
    return checkpoint
'''
