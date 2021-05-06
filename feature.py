'''
#  filename: feature.py
#  extract features of image patches using Resnet
#  Likun Qin, 2021
'''
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
from dataset.davis import Davis_dataset
from models.Resnet import resnet50, resnet101
from utils.image import resize_pieces


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
        val_dataset = Davis_dataset(data_root='/home/captain/dataset/DAVIS/DAVIS-semisupervised/DAVIS-trainval',
                                    size=(1600, 1600),
                                    slice=8,
                                    mode='val',
                                    transform=resize_pieces
                                    )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


    # create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if backbone == 'resnet50':
        model = resnet50(pretrained=True,
                         weight_path='/home/captain/projects/Video-inpainting/ckpt/resnet50-19c8e357.pth'
                         ).to(device)
    elif backbone == 'resnet101':
        model = resnet101(pretrained=True,
                         weight_path='/home/captain/projects/Video-inpainting/ckpt/resnet101-5d3b4d8f.pth'
                         ).to(device)

    # save some feature locally


    for batch, sample in enumerate(val_dataloader):
        if batch > 2:
            print(res.size())
            break
        img = sample['image'].to(device)
        img = torch.squeeze(img)
        res = model(img)



if __name__ == '__main__':
    extract_features(backbone='resnet50')


