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



class ComplementaryContentAttention(nn.Module):
    def __init__(self, num_dim=2048, head_dim=768, mid_dim=512):
        super().__init__()

        self.scale = (num_dim//2) ** -0.5
        self.num_dim=num_dim
        self.mid_dim = mid_dim

        self.linear_glo = nn.Sequential(nn.Linear(num_dim, mid_dim, bias=False),
                                        )
        self.linear_spa_1 = nn.Sequential(nn.Linear(num_dim, mid_dim, bias=False),
                                          )
        self.linear_spa_2 = nn.Sequential(nn.Linear(num_dim, mid_dim, bias=False),
                                          )
        self.linear_fusion = nn.Sequential(nn.Linear(head_dim, mid_dim, bias=False),
                                           )

        self.linear_v1 = nn.Sequential(nn.Linear(num_dim, mid_dim, bias=False),
                                           )
        self.linear_v2 = nn.Sequential(nn.Linear(num_dim, mid_dim, bias=False),
                                       )

        self.linear_project = nn.Sequential(nn.Linear(2*mid_dim, num_dim, bias=False),
                                            nn.ReLU(),
                                            nn.Linear(num_dim, 2*mid_dim, bias=False),
                                            )

        self.norm1 = nn.LayerNorm(num_dim)
        # self.norm2 = nn.LayerNorm(num_dim)

        # self.linear_learner = nn.Sequential(nn.Linear(num_dim, num_dim, bias=False),
        #                                     # nn.BatchNorm1d(num_dim),
        #                                     # nn.ReLU(),
        #                                     )

    def forward(self, x_spa, x_fusion):
        b = x_spa.size(0)
        n = x_spa.size(1)
        c = x_spa.size(2)

        x_glo = x_spa.mean(1).unsqueeze(dim=1)

        x_glo_norm = self.linear_glo(x_glo)
        x_fusion_norm = self.linear_fusion(x_fusion)
        x_spa_1_norm = self.linear_spa_1(x_spa)
        x_spa_2_norm = self.linear_spa_2(x_spa)

        spa_atte_1 = torch.matmul(x_glo_norm, x_spa_1_norm.transpose(1,2))
        spa_atte_2 = torch.matmul(x_fusion_norm, x_spa_2_norm.transpose(1, 2))

        spa_atte_sa = spa_atte_1 * self.scale
        spa_atte_sa = F.softmax(spa_atte_sa, dim=-1)

        spa_atte_ca = spa_atte_2 * self.scale
        spa_atte_ca = F.softmax(spa_atte_ca, dim=-1)

        x_spa_1 = self.linear_v1(x_spa)
        x_spa_2 = self.linear_v2(x_spa)

        x_1 = torch.matmul(spa_atte_sa, x_spa_1)
        x_2 = torch.matmul(spa_atte_ca, x_spa_2)

        x = torch.cat((x_1, x_2), dim=2) + x_glo

        x = self.linear_project(self.norm1(x)) + x

        return x



        # if self.num_dim ==2048:
        # if self.num_dim ==768:
        #     x = self.linear_learner(x.reshape(b, c)).unsqueeze(1)
        # return x
