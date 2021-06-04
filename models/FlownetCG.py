'''
#  filename: FlownetCG.py
#  implementation of graph based FlownetC
#  By Likun Qin ,2021
'''

import sys
sys.path.append('..')

from models.FlowNet2_Models.submodules import *
from models.FlowNet2_Models.correlation_package.correlation import Correlation
from models.GCN_model import GCN
# import cfgs.config_local as config
from cfgs import config
import torch.nn.init as init
import torch
import torch.nn as nn
import argparse
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from dataset.davis import FlownetCGData


class FlownetCG(nn.Module):
    '''
    incorporate FlowNet with GCN, addding features of nearby frames
    '''
    def __init__(self, batchnorm=False, batch_size=config.BATCH_SIZE):
        '''

        batchnorm: True or False, to add Batch Normalization or not
        @param batch_size: batch_size divided by gpu number, int
        @type batchnorm: object
        '''
        print('initiating FlowNetCG')
        super().__init__()

        print('initiating GCN')
        self.gcn = GCN(layers=3, frames=config.N, slice=config.SLICE, batch=batch_size)
        self.gcn = self.gcn.to('cuda')

        self.batchNorm = batchnorm

        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)

        self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2,
                                    corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        # self.fusion_conv = conv(self.batchNorm, in_planes=2048, out_planes=1024, kernel_size=1, stride=1)

        self.deconv5 = deconv(2048, 1024)
        self.deconv4 = deconv(1538, 768)
        self.deconv3 = deconv(1282, 512)
        self.deconv2 = deconv(770, 256)
        self.deconv1 = deconv(386, 128)

        self.predict_flow6 = predict_flow(2048)
        self.predict_flow5 = predict_flow(1538)
        self.predict_flow4 = predict_flow(1282)
        self.predict_flow3 = predict_flow(770)
        self.predict_flow2 = predict_flow(386)
        self.predict_flow1 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x, features):
        '''

        x: concatenated input adjacent frames, tensor, [N, 2C, H, W]
        features: features of continuous 2N frames, tensor, [N, nodes, 2048]

        x has shape of [N, 2C, H, W]
        generally C = 3
        '''
        x1 = x[:, 0:3, :, :]  # [N, C, H, W]
        x2 = x[:, 3::, :, :]  # [N, C, H, W]

        out_gcn = self.gcn(features)    # [N, 1024, 10, 10]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)     # [N, 128, 1/4 H, 1/4 W]
        out_conv3a = self.conv3(out_conv2a)     # [N, 256, 1/8 H, 1/8 W]

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)  # [N, 441, 1/8 H, 1/8 W]

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)  # [N, 32, 1/8 H, 1/8 W]

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)  # [N, 473, 1/8 H, 1/8 W]

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)  # [N, 256, 1/8 H, 1/8 W]

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))  # [N, 512, 1/16 H, 1/16 W]

        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # [N, 512, 1/32 H, 1/32 W]
        out_conv6 = self.conv6_1(self.conv6(out_conv5))  # [N, 1024, 1/64 H, 1/64 W]

        # for the case where H = 640, slice = 2
        out_fusion = torch.cat([out_conv6, out_gcn], dim=1)  # [N, 2048, 10, 10]

        flow6 = self.predict_flow6(out_fusion)  # [N, 2, 1/64 H, 1/64 W]
        flow6_up = self.upsampled_flow6_to_5(flow6)  # [N, 2, 1/32 H, 1/32 W]
        out_deconv5 = self.deconv5(out_fusion)  # [N, 1024, 1/32 H, 1/32 W]

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)  # [N, 1538, 1/32 H, 1/32 W]

        flow5 = self.predict_flow5(concat5)  # [N, 2, 1/32 H, 1/32 W]
        flow5_up = self.upsampled_flow5_to_4(flow5)  # [N, 2, 1/16 H, 1/16 W]
        out_deconv4 = self.deconv4(concat5)  # [N, 768, 1/16 H, 1/16 W]
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)  # [N, 1282, 1/16 H, 1/16 W]

        flow4 = self.predict_flow4(concat4)  # [N, 2, 1/16 H, 1/16 W]
        flow4_up = self.upsampled_flow4_to_3(flow4)  # [N, 2, 1/8 H, 1/8 W]
        out_deconv3 = self.deconv3(concat4)  # [N, 512, 1/8 H, 1/8 W]
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)  # [N, 770, 1/8 H, 1/8 W]

        flow3 = self.predict_flow3(concat3)  # [N, 2, 1/8 H, 1/8 W]
        flow3_up = self.upsampled_flow3_to_2(flow3)  # [N, 2, 1/4 H, 1/4 W]
        out_deconv2 = self.deconv2(concat3)  # [N, 256, 1/4 H, 1/4 W]
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)  # [N, 386, 1/4 H, 1/4 W]

        flow2 = self.predict_flow2(concat2)  # [N, 2, 1/4 H, 1/4 W]
        flow2_up = self.upsampled_flow2_to_1(flow2)  # [N, 2, 1/2 H, 1/2 W]
        out_deconv1 = self.deconv1(concat2)  # [N, 128, 1/2 H, 1/2 W]
        concat1 = torch.cat((out_conv1a, out_deconv1, flow2_up), 1)  # [N, 194, 1/2 H, 1/2 W]

        flow1 = self.predict_flow1(concat1)
        flow1_up = self.upsampled_flow2_to_1(flow1)

        if self.training:
            return flow1_up, flow1, flow2, flow3, flow4, flow5, flow6, out_conv3_1, out_conv4, out_conv5, out_conv6
        else:
            return flow1_up

    def fix_front(self):
        '''
        fix the weights of the upper part of the net
        @return:
        '''
        module_list = []
        for i, m in enumerate(self.modules()):
            # print(m)
            module_list.append(m)
            if i in range(7, 42):
                m.eval()
                for item in m.parameters(recurse=False):
                    item.requires_grad_(False)

        print('all!')


def change_state_dict(state_dict):
    '''
    for initialization of FlownetCG
    keep the first half of weights in original pretrained FlownetC
    delete the rest
    state_dict: OrderedDict, original weights for FlownetC
    '''
    state_dict = state_dict['state_dict']
    back = list(state_dict)[22:]
    for item in back:
        del state_dict[item]
    print('the rest half are deleted')
    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true')
    parser.parse_args()
    parser.fp16 = False

    ckpt = '/home/cap/project/inpainting/FlowNet2_checkpoint.pth.tar'
    checkpoint = torch.load(ckpt)
    truncated = change_state_dict(checkpoint)
    flownetcg = FlownetCG()
    # print(flownetcg)
    flownetcg.load_state_dict(truncated, strict=False)
    flownetcg.fix_front()
    flownetcg.to('cuda')
    print('model loaded')
    # for param in flownetcg.parameters():
    #    print(param)

    dataset = FlownetCGData(data_root=config.DATA_ROOT, mode='train')
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    for batch, data in enumerate(dataloader):
        frames = data['frames'].cuda()
        feature = data['feature'].cuda()
        res_flow = flownetcg(frames, feature)

        if batch == 0:
            break

    print('done!')





