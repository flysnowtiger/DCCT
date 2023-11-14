# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn


# from models.backbone.vision_transformer import vit_tiny_patch16_224_in21k, vit_small_patch16_224_in21k, deit_tiny_patch16_224
# import timm
import torchvision
from models.backbone.resnets1 import resnet34, resnet18
from models.backbone.vit_pytorch import TransReID
from models.backbone.vit_pytorch1 import TransReID1

from models.backbone.vit import ViT
import os.path as osp
from functools import partial

# ===================
#   Initialization
# ===================

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

class ViTBase(nn.Module):

    def __init__(self, num_classes, pretrained_model_dir, neck_feat = None, neck="no", **kwargs):
        super(ViTBase, self).__init__()

        self.num_classes = num_classes
        self.base_dim = 2048
        # '''#################################
        ### ViT-base Baseline
        pretrained_vit = osp.join(pretrained_model_dir, 'jx_vit_base_p16_224-80ecf9dd.pth')
        self.trans_backbone = TransReID(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=768, depth=12,
                                        num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                        camera=0, view=0, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.5, local_feature=False,
                                        hw_ratio=1, gem_pool=False, stem_conv=False)
        self.trans_backbone.load_param(pretrained_vit, hw_ratio=1)
        # self.trans_backbone.load_param('/home/omnisky/LXH/pretrain_model/vit_base_ics_cfs_lup.pth', hw_ratio=1)
        # self.trans_backbone_fc = nn.Sequential(nn.Linear(768, 2048))

        self.trans_base_dim = 768
        self.bottleneck_trans_base = nn.BatchNorm1d(self.trans_base_dim)
        self.bottleneck_trans_base.bias.requires_grad_(False)
        self.bottleneck_trans_base.apply(weight_init_kaiming)
        self.classifier_trans_base = nn.Linear(self.trans_base_dim, num_classes)
        self.classifier_trans_base.apply(weight_init_classifier)

        ########################################'''


    def forward(self, inputs, return_logits = False):

        b, t, c, h, w = inputs.size()
        v_input = inputs.view(b * t, c, h, w)  # 80, 3, 256, 128

        trans_cls_score_list = []
        trans_bn_feat_list = []

        # '''#######################################
        ### Transformer baseline
        x_trans = self.trans_backbone(v_input)

        f_trans_base = x_trans[:, 0, :]#.mean(1) # #
        # f_trans_base = self.trans_backbone_fc(x_trans[:, 0, :])
        f_trans_base = f_trans_base.reshape(b, t, -1).mean(1)
        BN_f_trans_base = self.bottleneck_trans_base(f_trans_base)
        trans_bn_feat_list.append(BN_f_trans_base)
        cls_score_trans_base = self.classifier_trans_base(BN_f_trans_base)
        trans_cls_score_list.append(cls_score_trans_base)
        #######################################'''

        if return_logits :
            return trans_cls_score_list

        if self.training:
            return  trans_cls_score_list, trans_bn_feat_list
        else:
            f_trans = torch.cat(trans_bn_feat_list, dim=1)
            return f_trans


