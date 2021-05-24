'''
#  filename: feature.py
#  extract features of image patches using Resnet
#  Likun Qin, 2021
'''
import pickle
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.davis import ResnetInfer
from models.Resnet import resnet50, resnet101
from cfgs import config

output_dir = '/mnt/qinlikun/inpainting/features'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

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
        feature_dataset = ResnetInfer(data_root='/mnt/qinlikun/dataset/tiny_DAVIS',
                                  mask_dir=None,
                                  out_dir=None,
                                  slice=config.SLICE,
                                  div=config.DIV,
                                  N=config.N
                                  )
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
        if batch == 0:
            print(sample['frames'][0].size())
            # print(sample['video'][0])

        for img in sample['frames']:
            # read in frames in a mini batch
            img = img.to(device)
            img = torch.squeeze(img)
            results.append(model(img))
            print(batch)

        # save features
        output_file = sample['out_file']
        print('output at ', output_file)
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)            # each pk file stores a list of tensor, which has size of [N, 2048]
        results = []                           # N = slice * slice / div


if __name__ == '__main__':
    extract_features(backbone='resnet50')


