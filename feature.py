'''
#  filename: feature.py
#  extract features of image patches using Resnet
#  Likun Qin, 2021
'''
import pickle
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.davis import Davis_dataset
from models.Resnet import resnet50, resnet101
from utils.image import resize_pieces

output_dir = '/mnt/qinlikun/inpainting/features'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def extract_features(backbone, dataset='davis', batch_size=1):
    '''
    extract feature
    :param backbone: str, 'resnet50' or 'resnet101'
    :param dataset: str, 'davis'
    :param batch_size: int
    :return:
    '''
    # load data
    if dataset == 'davis':
        val_dataset = Davis_dataset(data_root='/mnt/qinlikun/dataset/DAVIS',
                                    size=(2240, 2240),
                                    slice=10,
                                    mode='val',
                                    transform=resize_pieces
                                    )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


    # create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if backbone == 'resnet50':
        model = resnet50(pretrained=True,
                         weight_path='/mnt/qinlikun/inpainting/resnet/resnet50-19c8e357.pth'
                         )
    elif backbone == 'resnet101':
        model = resnet101(pretrained=True,
                         weight_path='/mnt/qinlikun/inpainting/resnet/resnet101-5d3b4d8f.pth'
                         )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        print('torch.cuda.device_count()')

    previous_video = None
    results = []

    for batch, sample in enumerate(val_dataloader):
        img = sample['image'].cuda()
        img = torch.squeeze(img)
        # res = model(img)
        if sample['video'] != previous_video and previous_video:
            with open(os.path.join(output_dir, previous_video + '.pk'), 'wb') as f:
                pickle.dump(results, f)
            results = []
            print(previous_video)
        results.append(model(img))
        previous_video = sample['video']

    with open(os.path.join(output_dir, previous_video + '.pk'), 'wb') as f:
        pickle.dump(results, f)







if __name__ == '__main__':
    extract_features(backbone='resnet50')


