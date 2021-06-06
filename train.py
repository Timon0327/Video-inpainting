'''
#  filename: train.py
#  train the FlownetCG
#  By Likun, 2021
'''
import torch
import os
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from dataset.davis import FlownetCGData
from models.FlownetCG import FlownetCG, change_state_dict
from cfgs import config
from utils.losses import L1Loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_flownet2', type=str,
                        default=config.FLOWNET_WEIGHT)
    parser.add_argument('--checkpoint', type=str,
                        default=None)

    parser.add_argument('--img_size', type=list, default=config.IMG_SIZE)
    parser.add_argument('--rgb_max', type=float, default=255.)

    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--LR', type=float, default=config.LR)

    parser.add_argument('--save_dir', type=str, default=config.SAVE_DIR)
    # parser.add_argument('--local_rank', action='store_true')

    parser.add_argument('--max_iter', type=int, default=config.MAX_ITER)
    parser.add_argument('--EPOCH', type=int, default=config.EPOCH)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=config.WEIGHTED_DECAY)
    parser.add_argument('--PRINT_EVERY', type=int, default=10)
    parser.add_argument('--MODEL_SAVE_STEP', type=int, default=1000)
    parser.add_argument('--DECAY_STEPS', type=list, default=config.DECAY_STEPS)

    # parser.add_argument('--VALID_EVERY', type=int, default=100)

    args = parser.parse_args()
    return args


def train(args):

    # set path
    log_dir = args.save_dir + '/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    ckpt_dir = args.save_dir + '/ckpt/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

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
    train_dataset = FlownetCGData(data_root=config.DATA_ROOT, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = FlownetCGData(data_root=config.DATA_ROOT, mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    valid_len = len(valid_dataset)

    # model
    gpu_num = torch.cuda.device_count()
    flownetcg = FlownetCG(batch_size=args.batch_size // gpu_num)
    # writer.add_graph(flownetcg)

    print("1 train epoch has ", len(train_dataset) // args.batch_size, ' iterations')
    print("1 valid epoch has ", len(valid_dataset) // args.batch_size, ' iterations')

    # loss and optimizer
    l1loss = L1Loss(args).to(device)
    optimizer = torch.optim.SGD(flownetcg.parameters(), lr=args.LR,
                                momentum=0.9, weight_decay=args.WEIGHT_DECAY)

    # whether to continue training or train from the start
    if args.checkpoint:
        # continue
        ckpt = torch.load(args.checkpoint)
        flownetcg.load_state_dict(ckpt['flownetcg'])
        step = ckpt['step']
        epc = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print('start from ', step, ' step')
    else:
        # start from scratch
        ckpt = change_state_dict(torch.load(args.pretrained_model_flownet2))
        flownetcg.load_state_dict(ckpt, strict=False)
        step = 0
        epc = 0
        print('start from the very begining')

    if torch.cuda.device_count() > 1:
        flownetcg = torch.nn.DataParallel(flownetcg)
        flownetcg = flownetcg.to(device)
        # flownetcg.module.gcn.to(device)
        # torch.distributed.init_process_group(backend="nccl",init_method='tcp://localhost:23456', rank=0, world_size=1)
        # flownetcg = flownetcg.to(device)
        # flownetcg.module.update_gcn_device(flownetcg.module.device)
        # flownetcg = DistributedDataParallel(flownetcg)
        print('using ', torch.cuda.device_count(), ' cuda device(s)')

    # best model
    best_step = 0
    best_epe = 100

    # train
    for epoch in range(epc, args.EPOCH):
        print('------------Next epoch ', epoch, '----------')
        print('training...')
        for data in train_dataloader:
            step += 1
            frames = data['frames'].to(device)
            feature = data['feature'].to(device)
            gt = data['gt'].to(device)

            res_flow = flownetcg(frames, feature)[0]
            if step == 1:
                print('frames size: ', frames.size())
                print('feature size: ', feature.size())
                print('gt size: ', gt.size())
                print('result size: ', res_flow.size())

            loss = l1loss(res_flow, gt)

            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()

            if step % args.PRINT_EVERY == 0:
                print('step: ', step)
                print('epoch: ', epoch)
                print(f"loss: {loss[0].item():>7f}")
                print(f"epe: {loss[1].item():>7f}")
                print('--------------------------------------')
                writer.add_scalar('loss', loss[0].item(), global_step=step)
                writer.add_scalar('epe', loss[1].item(), global_step=step)

            if step in args.DECAY_STEPS:
                optimizer['lr'] *= 0.1
                print('lr changed at step ', step)

            if step % args.MODEL_SAVE_STEP == 0:
                path = os.path.join(ckpt_dir, 'flownetcg_'+str(step)+'.pt')
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'flownetcg': flownetcg.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss[0].item()
                }, path)
                print('model has been saved in ', path)

        print('validation...')
        print('~~~~~~~~~~~~~~~~~~~~~')
        flownetcg.eval()
        test_losses = []
        with torch.no_grad():

            for j, valid_data in enumerate(valid_dataloader):
                print(j)
                frames = valid_data['frames'].to(device)
                feature = valid_data['feature'].to(device)
                gt = valid_data['gt'].to(device)
                res_flow = flownetcg(frames, feature)

                test_losses.append(l1loss(res_flow, gt)[1].item())
        test_epe = np.sum(test_losses) / valid_len
        print('epe for validation is ', test_epe)
        if test_epe < best_epe:
            best_epe = test_epe
            best_step = step
            print('new best!!')
        writer.add_scalar('valid epe', test_epe, global_step=step)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if step > args.max_iter:
            break

    path = os.path.join(ckpt_dir, 'flownetcg_' + 'final'+ str(step) + '.pt')
    torch.save({
        'epoch': epoch,
        'step': step,
        'flownetcg': flownetcg.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)
    print('model has been saved in ', path)

    print('best model at ', best_step)
    print('with epe at ', best_epe)

    writer.close()



if __name__ == '__main__':
    args = parse_args()
    train(args)




