import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import torch
import time
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset.davis import FlownetCGTest, ResnetInferTest
from models.Resnet import resnet50, resnet101
from models.FlownetCG import FlownetCG
from cfgs import config
import cvbase as cvb

from tools.frame_inpaint import DeepFillv1


def parse_argse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature', action='store_true')
    parser.add_argument('--flow', action='store_true')
    parser.add_argument('--dataset_root', type=str,
                        default=config.TEST_ROOT)
    parser.add_argument('--img_size', type=int, nargs='+',
                        default=config.IMG_SIZE)
    parser.add_argument('--rgb_max', type=float, default=255.)
    parser.add_argument('--frame_dir', type=str, default=None,
                        help='Give the dir of the video frames and generate the data list to extract flow')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Give the dir of the ground truth of video frames')
    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/qinlikun/inpainting/ckpt/flownetcg_24000.pt')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--output_root', type=str,
                        default=None)

    parser.add_argument('--DATA_ROOT', type=str,
                        default=None)

    parser.add_argument('--FIX_MASK', action='store_true')
    parser.add_argument('--enlarge_mask', action='store_true')
    parser.add_argument('--enlarge_kernel', type=int,
                        default=10)

    # Flow-Guided Propagation
    parser.add_argument('--th_warp', type=int, default=40)
    parser.add_argument('--img_root', type=str,
                        default=None)
    parser.add_argument('--mask_root', type=str,
                        default=None)
    parser.add_argument('--flow_root', type=str,
                        default=None)
    parser.add_argument('--output_root_propagation', type=str,
                        default=None)
    parser.add_argument('--pretrained_model_inpaint', type=str,
                        default=config.DEEPFILL_WEIGHT)

    args = parser.parse_args()

    return args


def extract_features(backbone, batch_size=1):
    '''
    extract feature
    :param backbone: str, 'resnet50' or 'resnet101'
    :param batch_size: int
    :return:
    '''
    # load data
    feature_dataset = ResnetInferTest(data_root=config.TEST_ROOT,
                                      mask_dir=None,
                                      out_dir=None,
                                      slice=config.SLICE,
                                      N=config.N
                                      )
    print(len(feature_dataset))
    dataloader = DataLoader(feature_dataset, batch_size=batch_size)

    # create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if backbone == 'resnet50':
        model = resnet50(pretrained=True,
                         weight_path=config.RESNET50_WEIGHT
                         )
    elif backbone == 'resnet101':
        model = resnet101(pretrained=True,
                         weight_path=config.RESNET101_WEIGHT
                         )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        print(torch.cuda.device_count())
    else:
        model = model.to(device)

    previous_video = None
    results = []

    for batch, sample in enumerate(dataloader):

        for img in sample['frames']:
            # read in frames in a mini batch
            img = img.to(device)
            img = torch.squeeze(img)
            res = model(img)
            results.append(res)
            print(batch)

        if batch == 0:
            print('input ', sample['frames'][0].size())
            # print(sample['video'][0])
            print('output:', res.size())

        # save features
        output_file = sample['out_file'][0]
        print('output at ', output_file)
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)            # each pk file stores a list of tensor, which has size of [N, 2048]
        results = []                           # N = slice * slice * 2N
    print("feature extraction completed!!")


def extract_flow(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using {} device".format(device))

    # dataset

    test_dataset = FlownetCGTest(data_root=config.TEST_ROOT)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    test_len = len(test_dataset)

    # model
    # gpu_num = torch.cuda.device_count()
    flownetcg = FlownetCG(batch_size=1)
    # writer.add_graph(flownetcg)

    print("It has ", len(test_dataset) // args.batch_size, ' iterations')
    print('validation set has ', len(test_dataset), ' image pairs')

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
    test_iterator = iter(test_dataloader)
    start_time = time.time()
    with torch.no_grad():

        for i in tqdm(range(0, len(test_dataset) // args.batch_size)):
            try:
                valid_data = next(test_iterator)
            except:
                print('Loader Restart')
                test_iterator = iter(test_dataloader)
                valid_data = next(test_iterator)

            # for j, valid_data in enumerate(valid_dataloader):
            # print(j)
            frames = valid_data['frames'].to(device)
            feature = valid_data['feature'].to(device)
            output_file = valid_data['out_file'][0]
            res_flow = flownetcg(frames, feature)

            res_flow = torch.squeeze(res_flow)
            result = res_flow.permute(1, 2, 0).data.cpu().numpy()
            cvb.write_flow(result, output_file)

    end_time = time.time()
    print('It takes ', end_time - start_time, ' seconds to get flows')
    output_dir = os.path.dirname(output_file)
    flow_list = [x for x in os.listdir(output_dir) if '.flo' in x]
    flow_start_no = min([int(x[:5]) for x in flow_list])

    zero_flow = cvb.read_flow(os.path.join(output_dir, flow_list[0]))
    cvb.write_flow(zero_flow*0, os.path.join(output_dir, '%05d.rflo' % flow_start_no))
    args.flow_root = output_file    # args.flow_root='.../demo/flow'
    print('Flow extraction completed!!')


def flow_guided_propagation(args):

    deepfill_model = DeepFillv1(pretrained_model=args.pretrained_model_inpaint,
                                image_shape=args.img_shape)

    from tools.propagation_inpaint import propagation
    propagation(args,
                frame_inapint_model=deepfill_model)


def psnr_and_ssim(args):
    '''
    calculate psnr and ssim
    @param args:
    @return:
    '''

    result_dir = os.path.join(args.output_root_propagation, 'inpaint_res')
    gt_dir = args.gt_dir

    result_list = os.listdir(result_dir)
    result_list.sort()

    psnrs = []
    ssims = []

    for one in result_list:
        name = one.split('.')[0]
        gt = cv.imread(os.path.join(gt_dir, name + '.jpg'))
        gt = cv.resize(gt, args.IMG_SIZE)
        pred = cv.imread(os.path.join(result_dir, one))

        psnr = peak_signal_noise_ratio(image_true=gt, image_test=pred.astype(np.uint8))
        ssim = structural_similarity(gt, pred.astype(np.uint8))

        print('image ', one)
        print('psnr: ', psnr, '      ssim: ', ssim)
        psnrs.append(psnr)
        ssims.append(ssim)

    psnr_mean = np.mean(psnrs)
    ssim_mean = np.mean(ssims)

    print('---------------------------')
    print('average psnr is ', psnr_mean)
    print('average ssim is ', ssim_mean)


def main():
    args = parse_argse()

    # if args.frame_dir = '.../demo/frames'

    if args.frame_dir is not None:
        args.dataset_root = os.path.dirname(args.frame_dir)  # args.dataset_root='.../demo"
    else:
        args.frame_dir = os.path.join(args.dataset_root, 'frames')

    if args.feature:
        extract_features(backbone='resnet50')

    if args.flow:
        extract_flow(args)   # args.flow_root='.../demo/flow'
    else:
        args.flow_root = os.path.join(args.dataset_root, 'flow')

    # set propagation args
    args.mask_root = os.path.join(args.dataset_root, 'masks')   # args.mask_root='.../demo/masks'
    args.img_root = args.frame_dir  # args.img_root='.../demo/frames'

    if args.output_root_propagation is None:
        args.output_root_propagation = os.path.join(args.dataset_root, 'Inpaint_Res')  # '.../demo/Inpaint_Res'
    if args.img_size is not None:
        args.img_shape = args.img_size

    flow_guided_propagation(args)

    if not args.gt_dir:
        args.gt_dir = args.frame_dir
    psnr_and_ssim(args)


if __name__ == '__main__':
    main()