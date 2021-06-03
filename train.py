'''
#  filename: train.py
#  train the FlownetCG
#  By Likun, 2021
'''
import torch
import os
import argparse
import yaml
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset.davis import GFCNetData
from models.FlownetCG import FlownetCG, change_state_dict
from cfgs import config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_flownet2', type=str,
                        default=config.FLOWNET_WEIGHT)
    parser.add_argument('--checkpoint', type=str,
                        default=None)

    parser.add_argument('--img_size', type=list, default=config.IMG_SIZE)
    parser.add_argument('--rgb_max', type=float, default=255.)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--max_iter', type=int, default=config.MAX_ITER)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=config.WEIGHTED_DECAY)
    parser.add_argument('--PRINT_EVERY', type=int, default=5)
    parser.add_argument('--MODEL_SAVE_STEP', type=int, default=5000)
    parser.add_argument('--NUM_ITERS_DECAY', type=int, default=10000)

    args = parser.parse_args()
    return args


def train(args):

    # set path
    log_dir = args.save_dir + '/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
        yaml.dump(vars(args), f)

    writer = SummaryWriter(log_dir=log_dir)

    # set device
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(7777777)
    else:
        device = "cpu"
        torch.manual_seed(7777777)

    print("Using {} device".format(device))

    # dataset
    dataset = GFCNetData(data_root=config.DATA_ROOT, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # model
    flownetcg = FlownetCG(args=args)

    # whether to continue training or train from the start
    if args.checkpoint:
        # continue
        ckpt = torch.load(args.checkpoint)
        flownetcg.load_state_dict(ckpt['flownetcg'])
    else:
        # start from scratch
        ckpt = change_state_dict(torch.load(args.pretrained_model_flownet2))
        flownetcg.load_state_dict(ckpt, strict=False)

    if torch.cuda.device_count() > 1:
        flownetcg= torch.nn.DataParallel(flownetcg)
        flownetcg = flownetcg.to(device)
        print('using ', torch.cuda.device_count(), ' cuda device(s)')







