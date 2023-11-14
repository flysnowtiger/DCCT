from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
import os.path as osp
import os
import torch.nn.functional as F
from functools import partial

import torchvision
from models.backbone.resnets1 import resnet34, resnet18, resnet50_s1
from models.backbone.vit_pytorch import TransReID
from models.backbone.resnet import *
from models.backbone.vit import ViT

from models.DCCT_KeyModule.CCA import ComplementaryContentAttention
from models.DCCT_KeyModule.HTA import HierachicalTemporalAggregation
from models.DCCT_KeyModule.SelfDistillation import KL_criterion


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




class DeepCoupledCnnTransformer(nn.Module):

    def __init__(self, num_classes, pretrained_model_dir, seq_len, model_mode, num_dim, layer, visual, **kwargs):
        super(DeepCoupledCnnTransformer, self).__init__()

        self.model_mode = model_mode
        self.HTA_layer = layer
        self.HTA_dim = num_dim

        self.visual = visual
        self.num_classes = num_classes

        if 'cnn' in self.model_mode:
            resnet50 = ResNet()
            init_pretrained_weight(resnet50, pretrained_path=pretrained_model_dir)
            print('Loading pretrained ImageNet model .........')
            self.cnn_backbone = nn.Sequential(resnet50.conv1,
                                        resnet50.bn1,
                                        resnet50.relu,
                                        resnet50.maxpool,
                                        resnet50.layer1,
                                        resnet50.layer2,
                                        resnet50.layer3,
                                        resnet50.layer4,
                                        )

            self.cnn_base_dim = 2048
            self.bottleneck_cnn_base = nn.BatchNorm1d(self.cnn_base_dim)
            self.classifier_cnn_base = nn.Linear(self.cnn_base_dim, self.num_classes, bias=False)
            self.bottleneck_cnn_base.apply(weight_init_kaiming)
            self.classifier_cnn_base.apply(weight_init_classifier)


        if 'transformer' in self.model_mode:

            pretrained_vit = osp.join(pretrained_model_dir, 'jx_vit_base_p16_224-80ecf9dd.pth')
            self.trans_backbone = TransReID(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=768, depth=12,
                                            num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                            camera=0, view=0, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.5, local_feature=False,
                                            hw_ratio=1, gem_pool=False, stem_conv=False)
            self.trans_backbone.load_param(pretrained_vit, hw_ratio=1)
            # self.trans_backbone.load_param('/home/omnisky/LXH/pretrain_model/vit_base_ics_cfs_lup.pth', hw_ratio=1)

            self.trans_base_dim = 768
            self.bottleneck_trans_base = nn.BatchNorm1d(self.trans_base_dim)
            self.bottleneck_trans_base.bias.requires_grad_(False)
            self.classifier_trans_base = nn.Linear(self.trans_base_dim, num_classes)
            self.bottleneck_trans_base.apply(weight_init_kaiming)
            self.classifier_trans_base.apply(weight_init_classifier)


        if 'cca' in self.model_mode:

            self.CCA_cnn = ComplementaryContentAttention(num_dim=self.cnn_base_dim, head_dim =self.trans_base_dim, mid_dim=self.cnn_base_dim//2)
            self.CCA_trans = ComplementaryContentAttention(num_dim=self.trans_base_dim, head_dim =self.cnn_base_dim, mid_dim=self.trans_base_dim//2)

            self.bottleneck_cnn_spa = nn.BatchNorm1d(self.cnn_base_dim)
            self.classifier_cnn_spa = nn.Linear(self.cnn_base_dim, self.num_classes, bias=False)
            self.bottleneck_cnn_spa.apply(weight_init_kaiming)
            self.classifier_cnn_spa.apply(weight_init_classifier)

            self.bottleneck_trans_spa = nn.BatchNorm1d(self.trans_base_dim)
            self.classifier_trans_spa = nn.Linear(self.trans_base_dim, self.num_classes, bias=False)
            self.bottleneck_trans_spa.apply(weight_init_kaiming)
            self.classifier_trans_spa.apply(weight_init_classifier)


        if 'hta' in self.model_mode:

            self.HTA = HierachicalTemporalAggregation(cnn_dim=self.cnn_base_dim, trans_dim = self.trans_base_dim,
                                                      num_dim=self.HTA_dim, layer=self.HTA_layer, num_tokens=seq_len)

            self.final_dim = self.HTA_dim*2
            self.bottleneck_final = nn.BatchNorm1d(self.final_dim)
            self.classifier_final = nn.Linear(self.final_dim, self.num_classes, bias=False)
            self.bottleneck_final.apply(weight_init_kaiming)
            self.classifier_final.apply(weight_init_classifier)


        if 'fd' in self.model_mode:
            self.Hint_cnn = nn.Sequential(nn.Linear(self.cnn_base_dim, self.final_dim))
            self.Hint_trans = nn.Sequential(nn.Linear(self.trans_base_dim, self.final_dim))


    def forward(self, inputs):

        b, t, c, h, w = inputs.size()
        v_input = inputs.view(b * t, c, h, w)  # 80, 3, 256, 128

        cls_score_list_base = []
        bn_feat_list_base = []

        cls_score_list = []
        bn_feat_list = []

    
        B = b
        T = t
        C = 2048
        H = 16
        W = 8

        ######### Base feature Supervision

        if 'cnn' in self.model_mode:
            x_cnn = self.cnn_backbone(v_input)
            cnn_feat_base = x_cnn.mean(-1).mean(-1)#.reshape(B, T, C).mean(1)
            bn_feature_1 = self.bottleneck_cnn_base(cnn_feat_base)
            cls_score_1 = self.classifier_cnn_base(bn_feature_1)
            cls_score_list_base.append(cls_score_1)
            bn_feat_list_base.append(bn_feature_1)

        # '''#######################################
        ### Transformer baseline

        if 'transformer' in self.model_mode:
            x_trans = self.trans_backbone(v_input)
            trans_feat_base = x_trans[:, 0, :]#.reshape(B, T, -1).mean(1)
            bn_feature_2 = self.bottleneck_trans_base(trans_feat_base)
            cls_score_2 = self.classifier_trans_base(bn_feature_2)
            bn_feat_list_base.append(bn_feature_2)
            cls_score_list_base.append(cls_score_2)
            #######################################'''

        ### Complementary Context Attention

        if 'cca' in self.model_mode:
            x_cnn_spa = self.CCA_cnn(x_cnn.flatten(2).transpose(1,2), x_trans[:,0,:].unsqueeze(dim=1).detach())
            x_trans_spa = self.CCA_trans(x_trans[:,1:,:], x_cnn.mean(-1).mean(-1).unsqueeze(dim=1).detach())

            cnn_feat_spa = x_cnn_spa.reshape(B, T, self.cnn_base_dim).mean(1)
            bn_feature = self.bottleneck_cnn_spa(cnn_feat_spa)
            cls_score = self.classifier_cnn_spa(bn_feature)
            cls_score_list.append(cls_score)
            bn_feat_list.append(bn_feature)

            trans_feat_spa = x_trans_spa.reshape(B, T, self.trans_base_dim).mean(1)
            bn_feature = self.bottleneck_trans_spa(trans_feat_spa)
            cls_score = self.classifier_trans_spa(bn_feature)
            cls_score_list.append(cls_score)
            bn_feat_list.append(bn_feature)

        ### Hierachical Temporal Aggregation

        if 'hta' in self.model_mode:
            x_cnn_temp = x_cnn_spa.detach().reshape(B, T, self.cnn_base_dim)
            x_trans_temp = x_trans_spa.detach().reshape(B, T, self.trans_base_dim)

            x_final = self.HTA(x_cnn_temp, x_trans_temp)

            feat_final = x_final.reshape(B, T, self.final_dim).mean(1)
            bn_feature_3 = self.bottleneck_final(feat_final)
            cls_score_3 = self.classifier_final(bn_feature_3)
            cls_score_list.append(cls_score_3)
            bn_feat_list.append(bn_feature_3)

        regular_loss = torch.tensor(0.0).cuda()

        if 'fd' in self.model_mode:

            cnn_feat =  self.Hint_cnn(x_cnn.mean(-1).mean(-1).reshape(B, T, -1))
            trans_feat =  self.Hint_trans(x_trans[:, 0, :].reshape(B, T, -1))
            feat_final = feat_final.unsqueeze(dim=1).expand(B, T,self.final_dim)

            regular_loss += torch.dist(cnn_feat, feat_final.detach()) + torch.dist(trans_feat, feat_final.detach())

        if 'ld' in self.model_mode:

            cnn_logit = cls_score_1
            trans_logit = cls_score_2
            final_logit = cls_score_3.unsqueeze(dim=1).expand(B, T, self.num_classes).reshape(B*T, self.num_classes)

            regular_loss += KL_criterion(cnn_logit, final_logit.detach()) + KL_criterion(trans_logit, final_logit.detach())


        if self.training:
                return  cls_score_list_base, bn_feat_list_base, cls_score_list, bn_feat_list, regular_loss
        else:
            f_test = []
            f1 = cnn_feat_base.reshape(B, T, self.cnn_base_dim).mean(1)
            f2 = trans_feat_base.reshape(B, T, self.trans_base_dim).mean(1)
            f_base = torch.cat((f1, f2), dim=1)
            f_test.append(f_base)

            f_spa = torch.cat((bn_feat_list[0], bn_feat_list[1]), dim=1)
            f_test.append(f_spa)

            f_final = bn_feat_list[-1]
            f_test.append(f_final)

            f_cat = torch.cat((f_base, f_spa, f_final), dim=1) ##
            f_test.append(f_cat)
            return f_test







        









