from __future__ import absolute_import
import os
import torch
from torch import nn
from  torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torchvision
from torch import nn

from models.backbone.resnet import *

__all__ = ['ResNet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def init_pretrained_weight(model,  pretrained_path):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    # pretrain_dict = model_zoo.load_url(model_url, model_dir = './')
    pretrain_model = os.path.join(pretrained_path, 'resnet50-19c8e357.pth')
    pretrain_dict = torch.load(pretrain_model)

    model_dict = model.state_dict()
    pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)

def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_out')
        nn.init.constant_(m.bias,0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight,1.0)
            nn.init.constant_(m.bias,0.0)

class ResNet50(nn.Module):

    def __init__(self, num_classes, model_name,  pretrain_choice, pretrained_model_dir, seq_len, neck_feat = None, neck="no", **kwargs):
        super(ResNet50,self).__init__()

        self.feat_dim = 2048
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.relu = nn.ReLU(inplace=True)

        self.base = ResNet()
        # self.down_channel = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0,
        #               bias=False),
        #     nn.BatchNorm2d(1024),
        #     self.relu
        # )
        
        if pretrain_choice == "imagenet":
            init_pretrained_weight(self.base, pretrained_path=pretrained_model_dir)
            print('Loading pretrained ImageNet model .........')


        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim,self.num_classes,bias=False)
        self.bottleneck.apply(weight_init_kaiming)
        self.classifier.apply(weight_init_classifier)

    def forward(self, x, pids = None, camids = None, return_logits = False):
        b,t,c,w,h = x.size()
        # print(x.size())
        x = x.view(b*t,c,w,h)
        feat = self.base(x)
        # feat = self.down_channel(feat)
        feat  = feat.view(b*t,self.feat_dim,feat.size(2),feat.size(3))
        feat = F.avg_pool2d(feat, feat.size()[2:])

        feat = feat.view(b, t, -1)
        feat = feat.mean(1)
        bn_feature = self.bottleneck(feat)
        cls_score = self.classifier(bn_feature)

        if return_logits :
            return cls_score

        if self.training:
            return cls_score,bn_feature
        else:
            return bn_feature



if __name__ == '__main__':
    x = torch.rand(16,5,3,256,128)
    num_classes = 625
    last_stride = 1
    model_path = '/home/wyq/.torch/models/resnet50-19c8e357.pth'
    neck = 'no'
    model_name = 'resnet50'
    pretrain_choice = 'imagenet'
    model = ResNet50(num_classes,last_stride,model_path,neck,model_name,pretrain_choice)
    model(x)