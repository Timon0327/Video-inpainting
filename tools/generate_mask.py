'''
#  filename: generate_mask.py
#  generate mask for corrupted video
'''
import numpy as np
import cv2 as cv
import os
import sys
sys.path.append('..')
from utils.image import rand_bbox_mask, mid_bbox_mask
from cfgs import config_ytb as config

data_dir = '/mnt/qinlikun/dataset/youtube/valid/JPEGImages'
# .../DAVIS*/JPEGImages/Full-Resolution or .../DAVIS*/JPEGImages/480p

save_dir = '/mnt/qinlikun/dataset/youtube/valid/masks'
# '/mnt/qinlikun/dataset/DAVIS/DAVIS-semisupervised/DAVIS-trainval/masks'

video_file = '/mnt/qinlikun/dataset/DAVIS/DAVIS-semisupervised/DAVIS-trainval/ImageSets/2017/train.txt'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if config.MASK_TYPE == 'mid':
    rect_shape = [int(x * config.MASK_RATIO) for x in config.IMG_SIZE]
    mask_channel = mid_bbox_mask(config.IMG_SIZE, rect_shape=rect_shape)
    zeros = np.zeros(config.IMG_SIZE)
    mask = np.concatenate([zeros[:, :, np.newaxis], zeros[:, :, np.newaxis], mask_channel[:, :, np.newaxis]], axis=2)
    cv.imwrite(os.path.join(save_dir, "mask.png"), mask)
else:
    if config.DATASET == 'davis':
        with open(video_file, 'r') as f:
            tmp = f.readlines()
            video_list = [x[:-1] for x in tmp]
    else:
        video_list = os.listdir(data_dir)
        try:
            t = video_list.index('.DS_Store')
            del video_list[t]
        except ValueError:
            pass
        video_list.sort()

    for vid in video_list:
        os.mkdir(os.path.join(save_dir, vid))

        img_list = os.listdir(os.path.join(data_dir, vid))
        img_size = config.IMG_SIZE
        ratio = config.MASK_RATIO
        rect_shape = [int(x * ratio) for x in img_size]

        for one in img_list:
            mask_channel = rand_bbox_mask(img_size, margin=(10, 10), rect_shape=rect_shape)

            zeros = np.zeros(img_size)
            mask = np.concatenate([zeros[:, :, np.newaxis], zeros[:, :, np.newaxis], mask_channel[:, :, np.newaxis]],
                                  axis=2)
            cv.imwrite(os.path.join(save_dir, vid, one[:-4] + '.png'), mask)


