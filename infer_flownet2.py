import sys, os, argparse
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import torch
import cvbase as cvb
from torch.utils.data import DataLoader
from mmcv import ProgressBar

from models import FlowNet2
from dataset.youtube import FlownetInfer
from cfgs import config_ytb as config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_flownet2', type=str,
                        default=config.FLOWNET_WEIGHT)
    parser.add_argument('--data_root', type=str,
                        default=config.VALID_ROOT)
    parser.add_argument('--mode', type=str, default='gt')
    parser.add_argument('--img_size', type=list, default=config.IMG_SIZE)
    parser.add_argument('--rgb_max', type=float, default=255.)
    parser.add_argument('--fp16', action='store_true')
    # parser.add_argument('--data_list', type=str, default=None, help='Give the data list to extract flow')
    # parser.add_argument('--frame_dir', type=str, default=None,
    #                     help='Give the dir of the video frames and generate the data list to extract flow')

    args = parser.parse_args()
    return args


def infer(args):

    device = torch.device('cuda')

    Flownet = FlowNet2(args, requires_grad=False)

    print('====> Loading', args.pretrained_model_flownet2)
    flownet2_ckpt = torch.load(args.pretrained_model_flownet2)
    Flownet.load_state_dict(flownet2_ckpt['state_dict'])

    if torch.cuda.device_count() > 1:
        Flownet = torch.nn.DataParallel(Flownet)
        Flownet = Flownet.to(device)
        print(torch.cuda.device_count())
    else:
        Flownet = Flownet.to(device)

    Flownet.eval()

    dataset_ = FlownetInfer(data_root=args.data_root,
                            out_dir=None
                            )
    dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=False)
    task_bar = ProgressBar(dataset_.__len__())

    for i, result in enumerate(dataloader_):

        f1 = result['frame1'].to(device)
        f2 = result['frame2'].to(device)      # [N, C ,H, W]

        print(f1.size())
        flow = Flownet(f1, f2)

        output_path = result['flow_file'][0]

        flow_numpy = flow[0].permute(1, 2, 0).data.cpu().numpy()
        cvb.write_flow(flow_numpy, output_path)
        task_bar.update()
    sys.stdout.write('\n')
    print('FlowNet2 Inference has been finished~!')
    # print('Extracted Flow has been save in', output_file)

    # return output_file


def main():
    args = parse_args()
    infer(args)


if __name__ == '__main__':
    main()
