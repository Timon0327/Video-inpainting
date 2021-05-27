'''
#  filename: image.py
#  functions to adjust images
#  Likun Qin, 2021
'''
import random
import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import ToTensor
totensor = ToTensor()


def apply_mask_resize(frame, size, slice, mask):
    '''
    split the image to slice * slice equal pieces
    :param frame: input image, ndarray, [H, W, C]
    :param size: the desired size of complete image, (height, width)
    :param slice: how many slices in each dimension, int
    :param mask: the mask, ndarray, [height, width]
    :return: Tensor, [slice*slice, C, H, W]
    '''
    img = cv.resize(frame, size)
    # check dimension
    assert img.shape[:-1] == mask.shape

    # apply mask
    img[mask > 0] = 0

    img_big = totensor(img).contiguous().float()
    imgs = []
    if slice > 1:
        for row in torch.chunk(img_big, slice, dim=1):
            for one in torch.chunk(row, slice, dim=2):
                imgs.append(one)
        return torch.stack(imgs, dim=0)
    else:
        return img_big



def apply_mask(frame, mask):
    '''
    set pixels in regions specified by mask to zero
    positions where mask value is larger than 0 are the ones to set
    :param frame: the image, ndarray, [height, width, channel]
    :param mask:  the mask, ndarray, [height, width]
    :return: fixed frame, ndarray, [height, width, channel]
    '''
    # check dimension
    assert frame.shape[:-1] == mask.shape

    frame[mask > 0] = 0

    return frame


def rand_bbox_mask(image_shape, margin, rect_shape):
    '''
    generate random rectangle mask
    :param image_shape: the shape of image, tuple, (height, width)
    :param margin: margins in each direction, tuple, (margin_y, margin_x)
    :param rect_shape: the shape of rectangle, tuple, (height, width)
    :return: mask, ndarray, [width, height]
    '''
    mask = np.zeros(image_shape)

    y = random.randint(margin[0], image_shape[0] - margin[0] - rect_shape[0])
    x = random.randint(margin[1], image_shape[1] - margin[1] - rect_shape[1])

    w = rect_shape[1]
    h = rect_shape[0]

    mask[y: y + h, x: x + w] = 128
    return mask


def image_and_mask(frame, size, slice, mask, margin, rect_shape):
    '''
    put together previous functions
    if mask is None, we will generate one
    :param frame: input image, ndarray, [H, W, C]
    :param size: the desired size of complete image, (height, width)
    :param slice: how many slices in each dimension, int
    :param mask: mask, ndarray, [height, width]; or None
    :param margin: margins in each direction, tuple, (margin_y, margin_x)
    :param rect_shape: the shape of mask, tuple, (height, width)
    :return:
    '''

    if not mask:
        mask = rand_bbox_mask(size, margin, rect_shape)

    img = apply_mask_resize(frame, size, slice, mask)

    return img, mask




