import sys, os
# sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import argparse
import yaml


import torch
import torch.optim as optim
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.loss as L
from models import resnet_models
from models.GCN_model import GCN
from utils.io import save_ckpt, load_ckpt
from utils.runner_func import *
from cfgs import config
from dataset.davis import GFCNetData


parser = argparse.ArgumentParser()

# training options
parser.add_argument('--save_dir', type=str, default='/mnt/qinlikun/inpainting')
# parser.add_argument('--model_name', type=str, default=None)

parser.add_argument('--max_iter', type=int, default=config.MAX_ITER)
parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
parser.add_argument('--n_threads', type=int, default=config.N_THREADS)
parser.add_argument('--resume', action='store_true')

parser.add_argument('--LR', type=float, default=config.LR)
parser.add_argument('--LAMBDA_SMOOTH', type=float, default=config.LAMBDA_SMOOTH)
parser.add_argument('--LAMBDA_HARD', type=float, default=config.LAMBDA_HARD)
# parser.add_argument('--BETA1', type=float, default=0.9)
# parser.add_argument('--BETA2', type=float, default=0.999)
parser.add_argument('--WEIGHT_DECAY', type=float, default=config.WEIGHTED_DECAY)

# parser.add_argument('--IMAGE_SHAPE', type=list, default=config.IMG_SIZE)  # [240, 424, 3]
# parser.add_argument('--RES_SHAPE', type=list, default=[240, 424, 3])
# parser.add_argument('--FIX_MASK', action='store_true')
# parser.add_argument('--MASK_MODE', type=str, default=None)
# parser.add_argument('--PRETRAINED', action='store_true')
# parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
parser.add_argument('--RESNET_PRETRAIN_MODEL', type=str,
                    default=config.RESNET50_WEIGHT)
# parser.add_argument('--TRAIN_LIST', type=str, default=None)
# parser.add_argument('--EVAL_LIST', type=str, default=None)
# parser.add_argument('--MASK_ROOT', type=str, default=None)
# parser.add_argument('--DATA_ROOT', type=str, default=None,
#                     help='Set the path to flow dataset')
# parser.add_argument('--INITIAL_HOLE', action='store_true')
# parser.add_argument('--TRAIN_LIST_MASK', type=str, default=None)

parser.add_argument('--PRINT_EVERY', type=int, default=5)
parser.add_argument('--MODEL_SAVE_STEP', type=int, default=5000)
parser.add_argument('--NUM_ITERS_DECAY', type=int, default=10000)
parser.add_argument('--CPU', action='store_true')

# parser.add_argument('--MASK_HEIGHT', type=int, default=120)
# parser.add_argument('--MASK_WIDTH', type=int, default=212)
# parser.add_argument('--VERTICAL_MARGIN', type=int, default=10)
# parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=10)
# parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=60)
# parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=106)

args = parser.parse_args()


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_save_dir = args.save_dir + '/ckpt/'
    sample_dir = args.save_dir + '/images/'
    log_dir = args.save_dir + '/log/'


    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
        yaml.dump(vars(args), f)

    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(7777777)
    if not args.CPU:
        torch.cuda.manual_seed(7777777)

    gcn_temporal = GCN(node_num=config.SLICE * config.SLICE * config.N,
                       layers=3)
    if torch.cuda.device_count() > 1:
        gcn_temporal = torch.nn.DataParallel(gcn_temporal)
        gcn_temporal = gcn_temporal.to(device)
        print(torch.cuda.device_count())

    flow_resnet = resnet_models.Flow_Branch_Multi(input_chanels=33, NoLabels=2)  # input_channel must be changed
    saved_state_dict = torch.load(args.RESNET_PRETRAIN_MODEL)
    for i in saved_state_dict:
        if 'conv1.' in i[:7]:
            conv1_weight = saved_state_dict[i]
            conv1_weight_mean = torch.mean(conv1_weight, dim=1, keepdim=True)
            conv1_weight_new = (conv1_weight_mean / 33.0).repeat(1, 33, 1, 1)
            saved_state_dict[i] = conv1_weight_new
    flow_resnet.load_state_dict(saved_state_dict, strict=False)
    flow_resnet = nn.DataParallel(flow_resnet).cuda()
    flow_resnet.train()

    optimizer = optim.SGD([{'params': get_1x_lr_params(flow_resnet.module), 'lr': args.LR},
                           {'params': get_10x_lr_params(flow_resnet.module), 'lr': 10 * args.LR}],
                          lr=args.LR, momentum=0.9, weight_decay=args.WEIGHT_DECAY)

    train_dataset = GFCNetData(data_root=config.DATA_ROOT,
                               mode='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.n_threads)

    if args.resume:
        if args.PRETRAINED_MODEL is not None:
            resume_iter = load_ckpt(args.PRETRAINED_MODEL,
                                    [('resnet', flow_resnet), ('gcn', gcn_temporal)],
                                    [('optimizer', optimizer)])
            print('Model Resume from', resume_iter, 'iter')
        else:
            print('Cannot load Pretrained Model')
            return
    else:
        resnet_ckpt = torch.load(args.RESNET_PRETRAIN_MODEL)
        flow_resnet.load_state_dict(resnet_ckpt, strit=False)
        print('Train GCN from scratch and finetune Resnet')
    # if args.PRETRAINED:
    #     if args.PRETRAINED_MODEL is not None:
    #         resume_iter = load_ckpt(args.PRETRAINED_MODEL,
    #                                 [('resnet', flow_resnet), ('gcn', gcn_temporal)],
    #                                 strict=True)
    #         print('Model Resume from', resume_iter, 'iter')

    train_iterator = iter(train_loader)

    loss = {}

    start_iter = 0 if not args.resume else resume_iter

    for i in tqdm(range(start_iter, args.max_iter)):
        # st = time.time()
        try:
            one = next(train_iterator)
        except:
            print('Loader Restart')
            train_iterator = iter(train_loader)
            one = next(train_iterator)

        # print(time.time()-st)
        features = one['feature'].cuda()
        flows = one['flows'].cuda()
        masks = one['masks'].cuda()
        gts = one['gts'].cuda()

        gcn_features = gcn_temporal(features)
        flow1x = flow_resnet(input_x)

        fake_flow = flow1x * mask[:,10:12,:,:] + flow_masked[:,10:12,:,:] * (1. - mask[:,10:12,:,:])
        loss['1x_recon'] = L.L1_mask(flow1x[:,:,:,:], gt_flow[:,10:12,:,:], mask[:,10:12,:,:])
        loss['1x_recon_hard'], new_mask = L.L1_mask_hard_mining(flow1x, gt_flow[:,10:12,:,:], mask[:,10:11,:,:])

        loss_total = loss['1x_recon'] + args.LAMBDA_HARD * loss['1x_recon_hard']

        if i % args.NUM_ITERS_DECAY == 0:
            adjust_learning_rate(optimizer, i, [30000, 50000])
            print('LR has been changed')

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if i % args.PRINT_EVERY == 0:
            print('=========================================================')
            print(args.model_name, "Rank[{}] Iter [{}/{}]".format(0, i + 1, args.max_iter))
            print('=========================================================')
            print_loss_dict(loss)
            write_loss_dict(loss, writer, i)

        if (i+1) % args.MODEL_SAVE_STEP == 0:
            save_ckpt(os.path.join(model_save_dir, 'DFI_%d.pth' % i),
                      [('resnet', flow_resnet), ('gcn', gcn_temporal)], [('optimizer', optimizer)], i)
            print('Model has been saved at %d Iters' % i)

    writer.close()


if __name__ == '__main__':
    main()
