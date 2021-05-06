'''
#  filename: image.py
#  functions to adjust images
#  Likun Qin, 2021
'''
import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import ToTensor
totensor = ToTensor()


def resize_pieces(frame, size, slice):
    '''
    split the image to slice * slice equal pieces
    :param frame: input image, ndarray, [H, W, C]
    :param size: the desired size of complete image, (height, width)
    :param slice: how many slices in each dimension, int
    :return: Tensor, [slice*slice, C, H, W]
    '''
    img = cv.resize(frame, size)
    img_big = totensor(img)
    imgs = []
    for row in torch.chunk(img_big, slice, dim=1):
        for one in torch.chunk(row, slice, dim=2):
            imgs.append(one)
    return torch.stack(imgs, dim=0)

