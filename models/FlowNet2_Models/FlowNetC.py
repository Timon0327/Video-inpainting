import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

# from .correlation_package.modules.correlation import Correlation
from .correlation_package.correlation import Correlation

from .submodules import *
'Parameter count , 39,175,298 '

class FlowNetC(nn.Module):
    def __init__(self,args, batchNorm=True, div_flow = 20):
        super(FlowNetC,self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow

        self.conv1   = conv(self.batchNorm,   3,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv_redir  = conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        if args.fp16:
            self.corr = nn.Sequential(
                tofp32(),
                Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1),
                tofp16())
        else:
            self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)

        self.corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)

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

    def forward(self, x):
        '''

        x has shape of [N, 2C, H, W]
        generally C = 3
        '''
        x1 = x[:,0:3,:,:]       # [N, C, H, W]
        x2 = x[:,3::,:,:]       # [N, C, H, W]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b) # False
        out_corr = self.corr_activation(out_corr)       # [N, 441, 1/8 H, 1/8 W]

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)        # [N, 32, 1/8 H, 1/8 W]

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)       # [N, 473, 1/8 H, 1/8 W]

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)      # [N, 256, 1/8 H, 1/8 W]

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))       # [N, 512, 1/16 H, 1/16 W]

        out_conv5 = self.conv5_1(self.conv5(out_conv4))     # [N, 512, 1/32 H, 1/32 W]
        out_conv6 = self.conv6_1(self.conv6(out_conv5))     # [N, 1024, 1/64 H, 1/64 W]

        flow6       = self.predict_flow6(out_conv6)     # [N, 2, 1/64 H, 1/64 W]
        flow6_up    = self.upsampled_flow6_to_5(flow6)      # [N, 2, 1/32 H, 1/32 W]
        out_deconv5 = self.deconv5(out_conv6)       # [N, 512, 1/32 H, 1/32 W]

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)     # [N, 1026, 1/32 H, 1/32 W]

        flow5       = self.predict_flow5(concat5)       # [N, 2, 1/32 H, 1/32 W]
        flow5_up    = self.upsampled_flow5_to_4(flow5)      # [N, 2, 1/16 H, 1/16 W]
        out_deconv4 = self.deconv4(concat5)     # [N, 256, 1/16 H, 1/16 W]
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)     # [N, 770, 1/16 H, 1/16 W]

        flow4       = self.predict_flow4(concat4)       # [N, 2, 1/16 H, 1/16 W]
        flow4_up    = self.upsampled_flow4_to_3(flow4)      # [N, 2, 1/8 H, 1/8 W]
        out_deconv3 = self.deconv3(concat4)     # [N, 128, 1/8 H, 1/8 W]
        concat3 = torch.cat((out_conv3_1,out_deconv3,flow4_up),1)       # [N, 386, 1/8 H, 1/8 W]

        flow3       = self.predict_flow3(concat3)       # [N, 2, 1/8 H, 1/8 W]
        flow3_up    = self.upsampled_flow3_to_2(flow3)      # [N, 2, 1/4 H, 1/4 W]
        out_deconv2 = self.deconv2(concat3)     # [N, 64, 1/4 H, 1/4 W]
        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)        # [N, 322, 1/4 H, 1/4 W]

        flow2 = self.predict_flow2(concat2)     # [N, 2, 1/4 H, 1/4 W]

        
        if self.training:
            return flow2,flow3,flow4,flow5,flow6,out_conv3_1,out_conv4,out_conv5,out_conv6
        else:
            return flow2,
        
        return flow2,flow3,flow4,flow5,flow6