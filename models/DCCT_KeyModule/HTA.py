import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable

import torchvision
import numpy as np

from functools import partial
from models.backbone.vision_transformer import VisionTransformer




class GatedModule(nn.Module):
    def __init__(self, num_dim):
        super().__init__()
        self.num_dim = num_dim

        self.linear_project = nn.Sequential(nn.Linear(num_dim*2, num_dim, bias=False))
        self.chanel_attention = nn.Sequential(nn.BatchNorm1d(num_dim),
                                                nn.Linear(num_dim, num_dim),
                                               nn.Sigmoid())


    def forward(self, x_temp, x_fusion):
        b = x_temp.size(0)
        n = x_temp.size(1)
        c = x_temp.size(2)

        x_fusion = self.linear_project(x_fusion)
        x_fusion = x_fusion.unsqueeze(dim=1).expand(b,n,c)

        c_atte = x_temp * x_fusion
        c_atte = self.chanel_attention(c_atte.reshape(b*n, c)).reshape(b, n, c)
        x_temp = x_temp*c_atte + x_fusion*(1-c_atte)
        # x_temp = x_temp + x_fusion
        return x_temp


class HierachicalTemporalAggregation(nn.Module):
    def __init__(self, cnn_dim, trans_dim, num_dim, layer, num_tokens=8):
        super().__init__()
        self.layer = layer

        self.align_projector_1 = nn.Sequential(nn.Linear(cnn_dim, num_dim))
        self.align_projector_2 = nn.Sequential(nn.Linear(trans_dim, num_dim))


        if layer >=1:
            self.cnn_TT_1 = VisionTransformer(
                token_num=num_tokens, patch_size=num_tokens, in_dim=num_dim, embed_dim=num_dim, depth=1, num_heads=8,
                # drop_path_rate = 0.1, drop_rate = 0.5, attn_drop_rate = 0.0,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

            self.trans_TT_1 = VisionTransformer(
                token_num=num_tokens, patch_size=num_tokens, in_dim=num_dim, embed_dim=num_dim, depth=1, num_heads=8,
                # drop_path_rate = 0.1, drop_rate = 0.5, attn_drop_rate = 0.0,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

            self.AT_1 = VisionTransformer(
                token_num=num_tokens, patch_size=num_tokens , in_dim=num_dim*2, embed_dim=num_dim*2, depth=1, num_heads=8,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        if layer >=2:
            self.cnn_GA_1 = GatedModule(num_dim=num_dim)
            self.trans_GA_1 = GatedModule(num_dim=num_dim)

            self.cnn_TT_2 = VisionTransformer(
                token_num=num_tokens, patch_size=num_tokens, in_dim=num_dim, embed_dim=num_dim, depth=1, num_heads=8,
                mlp_ratio=0.25, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

            self.trans_TT_2 = VisionTransformer(
                token_num=num_tokens, patch_size=num_tokens, in_dim=num_dim, embed_dim=num_dim, depth=1, num_heads=8,
                mlp_ratio=0.25, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

            self.AT_2 = VisionTransformer(
                token_num=num_tokens, patch_size=num_tokens, in_dim=num_dim*2, embed_dim=num_dim*2, depth=1, num_heads=8,
                mlp_ratio=0.25, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))


    def forward(self, x_cnn_temp, x_trans_temp):

        ## feature alignment
        x_cnn_temp = self.align_projector_1(x_cnn_temp)
        x_trans_temp = self.align_projector_2(x_trans_temp)

        #### First Block
        if self.layer >=1:
            x_cnn_temp = self.cnn_TT_1(x_cnn_temp, mode='pos')
            x_trans_temp = self.trans_TT_1(x_trans_temp, mode='pos')

            x_fusion_temp = torch.cat((x_cnn_temp, x_trans_temp), dim=2)
            x_fusion_temp = self.AT_1(x_fusion_temp, mode='pos')


        #### Second Block
        # '''
        if self.layer >=2:
            x_cnn_temp = self.cnn_GA_1(x_cnn_temp, x_fusion_temp.mean(1)) #
            x_trans_temp = self.trans_GA_1(x_trans_temp, x_fusion_temp.mean(1)) #

            x_cnn_temp = self.cnn_TT_2(x_cnn_temp, mode='pos')
            x_trans_temp = self.trans_TT_2(x_trans_temp, mode='pos')

            x_fusion_temp = torch.cat((x_cnn_temp, x_trans_temp), dim=2)

            x_fusion_temp = self.AT_2(x_fusion_temp, mode='pos')
        #####'''

        return x_fusion_temp