from torch import nn
from ops.basic_ops import ConsensusModule
from utils.transforms import *
from torch.nn.init import normal_, constant_
from scipy.ndimage import zoom
import cv2
import os, sys
from torch.cuda import amp


class VideoModel(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='BNInception',
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, crop_num=1, print_spec=True,
                 gsf=True, gsf_ch_ratio=100,
                 target_transform=None):
        super(VideoModel, self).__init__()
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.gsf = gsf
        self.gsf_ch_ratio = gsf_ch_ratio
        self.target_transform = target_transform

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if print_spec:
            print(("""
    Initializing Video Model with backbone: {}.
    Model Configurations:
                        GSF:                {}
                        Channel ratio:      {}
                        num_segments:       {}
                        consensus_module:   {}
                        dropout_ratio:      {}
            """.format(base_model, self.gsf, self.gsf_ch_ratio, self.num_segments, consensus_type, self.dropout)))

        self.feature_dim = self._prepare_base_model(base_model)
       
        self.feature_dim = self._prepare_model(num_class, self.feature_dim)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

    def _prepare_model(self, num_class, feature_dim):

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            if self.gsf:
                import backbones.resnetGSFModels as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True, num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                import torchvision.models.resnet as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True)
            # print(self.base_model)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        elif base_model == 'bninception':
            import backbones.pytorch_load as inception
            if self.gsf:
                    self.base_model = inception.BNInception_gsf(num_segments=self.num_segments,
                                                                gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.BNInception()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 1024
        elif base_model == 'inceptionv3':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
        return feature_dim

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        linear_weight = []
        linear_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for n, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear): 
                ps = list(m.parameters())
                linear_weight.append(ps[0])
                if len(ps) == 2:
                    linear_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                # if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': linear_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "linear_weight"},
            {'params': linear_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "linear_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input, with_amp=False, idx=0, target=0):
        with amp.autocast(enabled=with_amp):
            base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))

            if self.dropout > 0:
                base_out_logits = self.new_fc(base_out)

            if not self.before_softmax:
                base_out_logits = self.softmax(base_out_logits)
                
            base_out_logits = base_out_logits.view((-1, self.num_segments) + base_out_logits.size()[1:])

            output = self.consensus(base_out_logits)

        return output

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False,
                                                                         target_transform=self.target_transform)])
