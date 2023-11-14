from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import numpy as np
import torch
from sklearn.metrics import f1_score


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def disciminative(x):
    above_average = x >= np.min(x)
    r = np.zeros(above_average.shape[0])
    r[above_average] = 1 / np.sum(above_average)
    return r

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AttributesMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self, attr_num):
        self.attr_num = attr_num
        self.preds =  [[] for _ in range(attr_num)]
        self.gts = [[] for _ in range(attr_num)]
        self.acces = np.array([0 for _ in range(attr_num)])
        self.acces_avg = None
        self.f1_score_macros = None
        self.count = 0

    def update(self, preds, gts, acces, n):
        self.count += n
        self.acces += acces
        for i in range(len(preds)):
            self.preds[i].append(preds[i])
            self.gts[i].append(gts[i])

    def get_f1_and_acc(self, mean_indexes=None):
        if mean_indexes is None:
            mean_indexes = [_ for _ in range(self.attr_num)]
        if self.acces_avg is None:
            self.acces_avg = self.acces / self.count
        if self.f1_score_macros is None:
            self.f1_score_macros = np.array([f1_score(y_pred=self.preds[i], y_true=self.gts[i], average='macro') for i in [0, 1] + list(range(self.attr_num))])

        return self.f1_score_macros, self.acces_avg, np.mean(self.acces_avg[mean_indexes]), np.mean(self.f1_score_macros[mean_indexes])



def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def make_optimizer(cfg, model, mode=None):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer

    # param_ids = list(map(id, model.module.cnn_backbone.parameters())) \
    #             + list(map(id, model.module.base_layer_1.parameters())) \
    #             + list(map(id, model.module.base_layer_2.parameters())) \
    #             + list(map(id, model.module.base_layer_3.parameters())) \
    #             + list(map(id, model.module.base_layer_4.parameters())) #\
    #             # + list(map(id, model.module.base_layer_4_2.parameters()))
    # params_1 = [p for p in model.parameters() if id(p) in param_ids]
    #
    # param_ids = list(map(id, model.module.A_block_2.parameters())) \
    #             + list(map(id, model.module.A_block_3.parameters()))
    #
    # params_2 = [p for p in model.parameters() if id(p) in param_ids]
    #
    # param_ids = list(map(id, model.module.M_block_3.parameters())) \
    #             + list(map(id, model.module.M_block_4.parameters()))
    #
    # params_3 = [p for p in model.parameters() if id(p) in param_ids]
    #
    # param_ids = list(map(id, model.module.bottleneck_final.parameters())) \
    #             + list(map(id, model.module.classifier_final.parameters()))
    # params_4 = [p for p in model.parameters() if id(p) in param_ids]
    #
    # param_groups = [{'params': params_1, 'lr_mult': 1},
    #                  {'params': params_2, 'lr_mult': 0.1},
    #                  {'params': params_3, 'lr_mult': 1},
    #                  {'params': params_4, 'lr_mult': 1},
    #                  ]
    # if mode == 'addappe' or mode == 'addmotion' or mode == 'addall':
    #     optimizer = torch.optim.Adam(param_groups, lr=0.00035, weight_decay=0.0005)
    # elif mode == 'vit_our' or mode == 'vit_base':
    #     optimizer = torch.optim.SGD(param_groups, lr=0.01,
    #                                 momentum=0.9,
    #                                 weight_decay=5e-4,
    #                                 nesterov=True)
    # else:
    #     print('===========================ERROR======================No this model mode')

    # return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def DeepSupervision(criterion , xs, y , mode='CE'):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """

    loss = 0.
    if mode =='CE-frame':
        batch_size = y.size(0)
        for x in xs:
            len_frame = x.size(0) // batch_size
            x = x.reshape(batch_size, len_frame, x.size(1))
            for i in range(len_frame):
                loss += criterion(x[:,i,:], y)
            loss = loss / len_frame
        loss /= len(xs)

    if mode =='Trip-frame':
        batch_size = y.size(0)
        for x in xs:
            len_frame = x.size(0) // batch_size
            x = x.reshape(batch_size, len_frame, x.size(1))
            for i in range(len_frame):
                loss += criterion(x[:, i, :], y)
            loss = loss / len_frame
        loss /= len(xs)

    if mode =='CE-video':
        for x in xs:
            loss += criterion(x, y)
        loss /= len(xs)

    if mode =='Trip':
        for x in xs:
            loss += criterion(x, y)
        loss /= len(xs)

    return loss
