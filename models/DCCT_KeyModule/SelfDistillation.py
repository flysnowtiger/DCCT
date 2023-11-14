import time
import torch
from torch import nn
import torch.nn.functional as F


def KL_criterion(outputs, targets):
    # temperature = 0.3
    # log_softmax_outputs = F.log_softmax(outputs / temperature, dim=1)
    # softmax_targets = F.softmax(targets / temperature, dim=1)
    # kl_loss = -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
    # print(kl_loss)

    T = 0.3
    kl_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                  F.softmax(targets / T, dim=1))
    # print(kl_loss)
    return kl_loss