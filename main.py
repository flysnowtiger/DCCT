from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

from torch.utils.data import DataLoader

import data_manager
from samplers import RandomIdentitySampler
# from video_loader import VideoDataset
# from video_loader_LMDB import VideoDatasetLMDB
from Data2LMDB import DatasetLMDB

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from thop import profile

from lr_schedulers import WarmupMultiStepLR
import transforms as T
import models
from losses import TripletLoss
from utils import AverageMeter, Logger, make_optimizer, DeepSupervision
from eval_metrics import evaluate_reranking
from config import cfg
# from ptflops import get_model_complexity_info


import warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument("--config_file", default="./configs/softmax_triplet.yml", help="path to config file", type=str)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)

parser.add_argument('--train_sampler', type=str, default='Random_interval', help='train sampler', choices=['Random_interval','Random_choice'])
parser.add_argument('--test_sampler', type=str, default='Begin_interval', help='test sampler', choices=['dense', 'Begin_interval'])
parser.add_argument('--triplet_distance', type=str, default='cosine', choices=['cosine','euclidean'])
parser.add_argument('--test_distance', type=str, default='cosine', choices=['cosine','euclidean'])
parser.add_argument('--split_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='mars', choices=['mars','duke', 'lsvid'])
parser.add_argument('--seq_len', type=int, default=8)

parser.add_argument('--arch', type=str, default='DCCT')
parser.add_argument('--gpu_device', type=str, default="6,7")
parser.add_argument('--method_name', type=str, default='Debug')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--model_mode', nargs='+', type=str, default=['cnn','transformer','cca','hta'])  #,'fd','ld'
parser.add_argument('--layer', type=int, default=2)
parser.add_argument('--num_dim', type=int, default=512)
parser.add_argument('--changed_thing', type=str, default='None')

parser.add_argument('--visual', action='store_true', default=False)
parser.add_argument('--only_test', action='store_true', default=False) ##
parser.add_argument('--test_path', type=str, default='')

parser.add_argument("--log_dir", default="/home/omnisky/LXH/projects/log_DCCT/", type=str)
parser.add_argument("--data_dir", default="/home/omnisky/LXH/data/", type=str)
# parser.add_argument("--data_dir", default="/media/omnisky/Data/LXH/LXH-20221002/LXH/LXH-20220908/data/LS-VID_V2/", type=str)
parser.add_argument("--model_dir", default="/home/omnisky/LXH/pretrain_model/", type=str)

parser.add_argument('--istation', action='store_true', default=False)
parser.add_argument("--server_log_dir", default="/17739334165/LXH_iStation/Project/log_DCCT", type=str)
parser.add_argument("--server_data_dir", default="/17739334165/LXH_iStation/Data", type=str)
parser.add_argument("--server_model_dir", default="/17739334165/LXH_iStation/PretrainedModel", type=str)

parser.add_argument('--bitahub', action='store_true', default=False)
parser.add_argument("--bitahub_log_dir_mars", default="/data/snowtiger/MARS/log_DCCT", type=str)
parser.add_argument("--bitahub_data_dir", default="/data/snowtiger", type=str)
parser.add_argument("--bitahub_model_dir", default="/data/snowtiger/PretrainedModel", type=str)

####   khahaqhahh  ####
print(os.getcwd())
user_name = os.getcwd().split('/')[1]
args_ = parser.parse_args()

if args_.bitahub == True:
    args_.config_file = "./PPL_VideoReID_CE_Trip/configs/softmax_triplet.yml"
if args_.istation == True:
    args_.config_file = "/17739334165/LXH_iStation/Project/PPL_VideoReID_CE_Trip_iStation/configs/softmax_triplet.yml"

if args_.config_file != "":
    cfg.merge_from_file(args_.config_file)
cfg.merge_from_list(args_.opts)

tqdm_enable = False

def main():
    if args_.istation == True:
        args_.gpu_device = '0,1'
        args_.log_dir = args_.server_log_dir
        args_.data_dir = args_.server_data_dir
        args_.model_dir = args_.server_model_dir

    if args_.bitahub == True:
        if args_.dataset == 'mars':
            args_.log_dir = args_.bitahub_log_dir_mars
        args_.data_dir = args_.bitahub_data_dir
        args_.model_dir = args_.bitahub_model_dir
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args_.istation == False and args_.bitahub == False:
        os.environ['CUDA_VISIBLE_DEVICES'] = args_.gpu_device

    args_.log_dir = os.path.join(args_.log_dir, args_.dataset)
    if not os.path.exists(args_.log_dir):
        os.mkdir(args_.log_dir)
    args_.log_dir = os.path.join(args_.log_dir, args_.method_name)
    if not os.path.exists(args_.log_dir):
        os.mkdir(args_.log_dir)
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    runId = args_.changed_thing + runId
    args_.log_dir = os.path.join(args_.log_dir, runId)
    if not os.path.exists(args_.log_dir):
        os.mkdir(args_.log_dir)
    print(args_.log_dir)
    if args_.only_test:
        test_save_dir = os.path.dirname(args_.test_path)
        sys.stdout = Logger(osp.join(test_save_dir, 'log_test.txt'))
    else:
        sys.stdout = Logger(osp.join(args_.log_dir, 'log_train.txt'))

    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    print("=========================\nConfigs:{}\n=========================".format(cfg))
    s = str(args_).split(", ")
    print("Fine-tuning detail:")
    for i in range(len(s)):
        print(s[i])
    print("=========================")

    # os.environ['CUDA_VISIBLE_DEVICES'] = args_.gpu_device  # cfg.MODEL.DEVICE_ID
    use_gpu = torch.cuda.is_available() and cfg.MODEL.DEVICE == "cuda"
    if use_gpu:
        if args_.bitahub == False:
            print("Currently using GPU {}".format(args_.gpu_device))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    if args_.bitahub == False:
        print("Initializing dataset {}".format(cfg.DATASETS.NAME))
        dataset = data_manager.init_dataset(root=args_.data_dir, name=args_.dataset, split_id = args_.split_id)
        dataset_num_train_pids = dataset.num_train_pids
        dataset_train = dataset.train
        dataset_query = dataset.query
        dataset_gallery = dataset.gallery
        from video_loader import VideoDataset

    else:
        if args_.dataset == 'mars':
            dataset_num_train_pids = 625
            train_dir = os.path.join(args_.data_dir,'MARS', '{}_train.lmdb'.format(args_.dataset))
            query_dir = os.path.join(args_.data_dir, 'MARS', '{}_query.lmdb'.format(args_.dataset))
            gallery_dir = os.path.join(args_.data_dir, 'MARS', '{}_gallery.lmdb'.format(args_.dataset))
        if args_.dataset == 'duke':
            dataset_num_train_pids = 702
            train_dir = os.path.join(args_.data_dir, 'DukeMCMTVID', '{}_train.lmdb'.format(args_.dataset))
            query_dir = os.path.join(args_.data_dir, 'DukeMCMTVID', '{}_query.lmdb'.format(args_.dataset))
            gallery_dir = os.path.join(args_.data_dir, 'DukeMCMTVID', '{}_gallery.lmdb'.format(args_.dataset))
        dataset_train = DatasetLMDB(train_dir)
        dataset_query = DatasetLMDB(query_dir)
        dataset_gallery = DatasetLMDB(gallery_dir)
        from video_loader_LMDB import VideoDataset

    if args_.visual:
        transform_train = T.Compose([
            T.resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.to_tensor(),
            T.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_train = T.Compose([
            T.resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.to_tensor(),
            T.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.random_erasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    transform_test = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pin_memory = True if use_gpu else False
    video_sampler = RandomIdentitySampler(dataset_train, num_instances=cfg.DATALOADER.NUM_INSTANCE)

    trainloader = DataLoader(
        VideoDataset(dataset_train, seq_len=args_.seq_len, sample=args_.train_sampler, transform=transform_train,
                     dataset_name=args_.dataset),
        sampler=video_sampler,
        batch_size=args_.batch_size, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=True
    )

    print('Build dense sampler')
    queryloader_dense = DataLoader(
        VideoDataset(dataset_query, seq_len=args_.seq_len, sample='dense', transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
        batch_size=1 , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader_dense = DataLoader(
        VideoDataset(dataset_gallery, seq_len=args_.seq_len, sample='dense', transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
        batch_size=1 , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False,
    )

    queryloader = DataLoader(
        VideoDataset(dataset_query, seq_len=args_.seq_len, sample='Begin_interval',
                     transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
        batch_size=cfg.TEST.SEQS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset_gallery, seq_len=args_.seq_len, sample='Begin_interval',
                     transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
        batch_size=cfg.TEST.SEQS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args_.arch))

    model = models.init_model(name=args_.arch, num_classes=dataset_num_train_pids, pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
                              model_name=cfg.MODEL.NAME, seq_len = args_.seq_len,
                              pretrained_model_dir=args_.model_dir,
                              model_mode=args_.model_mode, num_dim=args_.num_dim, layer=args_.layer, visual= args_.visual)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    inputs = torch.randn(1, 8, 3, 256, 128)
    flops, params = profile(model, (inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M') ### (12*n*(d**2) + 2*(n**2)*d)*4

    # n= 4
    # t = 8
    # d = 512
    # result = 2*((n*d)**2) + n*(4*t*(d**2) + 2*(t**2)*d)
    # result = (result * 2)/ (1e9)
    # print (result)

    model = nn.DataParallel(model)
    model.cuda()

    if args_.only_test:
        print("Loading checkpoint from '{}'".format(args_.test_path))
        print("load model... ")
        checkpoint = torch.load(args_.test_path)
        model.load_state_dict(checkpoint)
        print("this method: {:}".format(args_.method_name))
        # print("==> Interval Test")
        # _, metrics = test(model, queryloader, galleryloader, use_gpu)
        print("==> Dense Test")
        _, metrics = test(model, queryloader_dense, galleryloader_dense, use_gpu)
    else:
        start_time = time.time()
        xent = nn.CrossEntropyLoss()
        # xent = CrossEntropyLabelSmooth(num_classes=dataset_num_train_pids)
        tent = TripletLoss(cfg.SOLVER.MARGIN, distance=args_.triplet_distance)

        optimizer = make_optimizer(cfg, model, mode=args_.model_mode) ##, optimizer_sgd
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        # scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)

        start_epoch = 0
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
            if args_.visual:
                print("Loading checkpoint from '{}'".format(args_.test_path))
                print("load model... ")
                checkpoint = torch.load(args_.test_path)
                model.load_state_dict(checkpoint)
                train(model, trainloader, xent, tent, optimizer, use_gpu, args_.visual)  ##

            # _, metrics = test(model, queryloader, galleryloader, use_gpu)

            print("==> Epoch {}/{}".format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
            print("current lr:", scheduler.get_lr()[0])

            train(model, trainloader, xent, tent, optimizer, use_gpu, args_.model_mode, args_.visual)  ##
            scheduler.step()
            torch.cuda.empty_cache()

            if cfg.SOLVER.EVAL_PERIOD > 0 and ((epoch + 1) % cfg.SOLVER.EVAL_PERIOD == 0 or (epoch + 1) == cfg.SOLVER.MAX_EPOCHS) or epoch == 0 or epoch==9:
                print("==> Test")
                print("this method: {:}".format(args_.method_name))
                print("changed thing: {:}".format(args_.changed_thing))

                _, metrics = test(model, queryloader, galleryloader, use_gpu)
                rank1 = metrics[0]
                if epoch>220:
                    state_dict = model.state_dict()
                    torch.save(state_dict, osp.join(args_.log_dir, "rank1_" + str(rank1) + '_checkpoint_ep' + str(epoch + 1) + '.pth'))
                    if (epoch + 1) == 350 or (epoch + 1) == cfg.SOLVER.MAX_EPOCHS:
                        print("==> Dense Test")
                        print("This method is: {}".format(args_.method_name))
                        _, metrics = test(model, queryloader_dense, galleryloader_dense, use_gpu)

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(model, trainloader, xent, tent, optimizer, use_gpu, model_mode, visual):  ## optimizer_sgd,

    model.train()
    xent_losses_frame = AverageMeter()
    tent_losses_frame = AverageMeter()
    xent_losses = AverageMeter()
    tent_losses = AverageMeter()
    regular_losses = AverageMeter()
    losses = AverageMeter()
    lambda_1 = 1

    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):

        if use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()


        outputs_base, features_base, outputs, features, regular_loss = model(imgs)  #


        if isinstance(outputs_base, (tuple, list)):
            xent_loss_frame = DeepSupervision(xent, outputs_base, pids, mode='CE-frame')
        else:
            xent_loss_frame = xent(outputs, pids)

        if isinstance(features_base, (tuple, list)):
            tent_loss_frame = DeepSupervision(tent, features_base, pids, mode='Trip-frame')
        else:
            tent_loss_frame = tent(features, pids)

        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(xent, outputs, pids, mode='CE-video')
        else:
            xent_loss = xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            tent_loss = DeepSupervision(tent, features, pids, mode='Trip')
        else:
            tent_loss = tent(features, pids)

        xent_losses.update(xent_loss.item(), 1)
        tent_losses.update(tent_loss.item(), 1)
        xent_losses_frame.update(xent_loss.item(), 1)
        tent_losses_frame.update(tent_loss.item(), 1)
        regular_losses.update(lambda_1*regular_loss.mean().item(), 1)

        loss = xent_loss_frame + tent_loss_frame + xent_loss + tent_loss + lambda_1*regular_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), 1)
##
    print("Batch {}/{}\t Loss {:.6f} , cont_losses.val, cont_losses.avg({:.6f}) xent Loss {:.6f} ({:.6f}), tent Loss {:.6f} ({:.6f}), regular Loss {:.6f} ({:.6f})".format(
        batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, tent_losses.val,
        tent_losses.avg, regular_losses.val, regular_losses.avg))
    return losses.avg


def test(model, queryloader, galleryloader, use_gpu, ranks=[1,5,10,20]):
    K=4
    for k in range(0,K):
        with torch.no_grad():
            model.eval()
            qf, q_pids, q_camids =  [], [], []
            query_pathes = []
            for batch_idx, (imgs, pids, camids, img_path) in enumerate(queryloader): ##tqdm(
                query_pathes.append(img_path[0])
                del img_path
                if use_gpu:
                    imgs = imgs.cuda()
                    pids = pids.cuda()
                    camids = camids.cuda()

                if len(imgs.size()) == 6:
                    method = 'dense'
                    b, n, s, c, h, w = imgs.size()
                    assert (b == 1)
                    imgs = imgs.view(b * n, s, c, h, w)
                else:
                    method = None

                features = model(imgs)
                q_pids.extend(pids.data.cpu())
                q_camids.extend(camids.data.cpu())

                features = features[k].data.cpu()
                torch.cuda.empty_cache()
                features = features.view(-1, features.size(1))

                if method == 'dense':
                    features = torch.mean(features, 0,keepdim=True)
                qf.append(features)

            qf = torch.cat(qf,0)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)
            np.save("query_pathes", query_pathes)

            print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

            gf, g_pids, g_camids = [], [], []
            gallery_pathes = []
            for batch_idx, (imgs, pids, camids, img_path) in enumerate(galleryloader): ##tqdm(
                gallery_pathes.append(img_path[0])
                if use_gpu:
                    imgs = imgs.cuda()
                    pids = pids.cuda()
                    camids = camids.cuda()

                if len(imgs.size()) == 6:
                    method = 'dense'
                    b, n, s, c, h, w = imgs.size()
                    assert (b == 1)
                    imgs = imgs.view(b * n, s, c, h, w)
                else:
                    method = None

                features = model(imgs)
                features = features[k].data.cpu()
                torch.cuda.empty_cache()
                features = features.view(-1, features.size(1))

                if method == 'dense':
                    features = torch.mean(features, 0, keepdim=True)

                g_pids.extend(pids.data.cpu())
                g_camids.extend(camids.data.cpu())
                gf.append(features)

            gf = torch.cat(gf,0)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)

            if args_.dataset == 'mars':
                # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
                gf = torch.cat((qf, gf), 0)
                g_pids = np.append(q_pids, g_pids)
                g_camids = np.append(q_camids, g_camids)

            np.save("gallery_pathes", gallery_pathes)
            print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
            print("Computing distance matrix")

            be_cmc, metrics = evaluate_reranking(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, args_.test_distance)
    return metrics, be_cmc

if __name__ == '__main__':

    main()




