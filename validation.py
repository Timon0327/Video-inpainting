'''
#  filename: validation.py
#  Likun Qin, 2021
'''

import torch
import time
import argparse
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from dataset.davis import FlownetCGTrain
from models.FlownetCG import FlownetCG
from cfgs import config
from utils.losses import L1Loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str,
                        default=None)

    parser.add_argument('--img_size', type=list, default=config.IMG_SIZE)
    parser.add_argument('--rgb_max', type=float, default=255.)

    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)

    parser.add_argument('--save_dir', type=str, default=config.SAVE_DIR)
    # parser.add_argument('--local_rank', action='store_true')

    # parser.add_argument('--VALID_EVERY', type=int, default=100)

    args = parser.parse_args()
    return args


def valid(args):
    # set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using {} device".format(device))

    # dataset

    valid_dataset = FlownetCGTrain(data_root=config.DATA_ROOT, mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    valid_len = len(valid_dataset)

    # model
    gpu_num = torch.cuda.device_count()
    flownetcg = FlownetCG(batch_size=args.batch_size // gpu_num)
    # writer.add_graph(flownetcg)

    print("1 valid epoch has ", len(valid_dataset) // args.batch_size, ' iterations')
    print('validation set has ', len(valid_dataset), ' image pairs')

    # loss and optimizer
    l1loss = L1Loss(args).to(device)

    # whether to continue training or train from the start

    # continue
    assert args.checkpoint != None
    ckpt = torch.load(args.checkpoint)
    flownetcg.load_state_dict(ckpt['flownetcg'])
    step = ckpt['step']
    epc = ckpt['epoch']
        # optimizer.load_state_dict(ckpt['optimizer'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.cuda()
    print('model from ', step, ' step')

    if torch.cuda.device_count() > 1:
        flownetcg = torch.nn.DataParallel(flownetcg)
        flownetcg = flownetcg.to(device)
        # flownetcg.module.gcn.to(device)
        # torch.distributed.init_process_group(backend="nccl",init_method='tcp://localhost:23456', rank=0, world_size=1)
        # flownetcg = flownetcg.to(device)
        # flownetcg.module.update_gcn_device(flownetcg.module.device)
        # flownetcg = DistributedDataParallel(flownetcg)
        print('using ', torch.cuda.device_count(), ' cuda device(s)')

    flownetcg.eval()
    epes = []
    valid_iterator = iter(valid_dataloader)
    start_time = time.time()
    with torch.no_grad():

        for i in tqdm(range(0, len(valid_dataset) // args.batch_size)):
            try:
                valid_data = next(valid_iterator)
            except:
                print('Loader Restart')
                valid_iterator = iter(valid_dataloader)
                valid_data = next(valid_iterator)

            # for j, valid_data in enumerate(valid_dataloader):
            # print(j)
            frames = valid_data['frames'].to(device)
            feature = valid_data['feature'].to(device)
            gt = valid_data['gt'].to(device)
            res_flow = flownetcg(frames, feature)
            if i == 1:
                print('gt size ', gt.size())
                print('result size', res_flow.size())
            epes.append(l1loss(res_flow, gt)[1].item())
    end_time = time.time()
    print('how many epe: ', len(epes))
    print('epes[0]:', epes[0])
    test_epe = np.sum(epes) / len(epes)
    print('epe for validation is ', test_epe)
    print('cost time: ', end_time - start_time)


if __name__ == '__main__':
    args = parse_args()
    valid(args)


