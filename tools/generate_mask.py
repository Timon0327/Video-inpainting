'''
#  filename: generate_mask.py
#  generate mask for corrupted video
'''
import numpy as np
import cv2 as cv
import os
from utils.image import rand_bbox_mask
from cfgs import config

data_dir = '/home/captain/dataset/tiny_DAVIS/JPEGImages'
# .../DAVIS*/JPEGImages/Full-Resolution or .../DAVIS*/JPEGImages/480p

save_dir = '/home/captain/dataset/tiny_DAVIS/masks'

video_file = '/home/captain/dataset/tiny_DAVIS/ImageSets/2017/train.txt'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

with open(video_file, 'r') as f:
    tmp = f.readlines()
    video_list = [x[:-1] for x in tmp]

for vid in video_list:
    os.mkdir(os.path.join(save_dir, vid))

    img_list = os.listdir(os.path.join(data_dir, vid))
    img_size = config.IMG_SIZE
    ratio = config.MASK_RATIO
    rect_shape = [int(x * ratio) for x in img_size]

    for one in img_list:
        mask_channel = rand_bbox_mask(img_size, margin=(10, 10), rect_shape=rect_shape)
        zeros = np.zeros(img_size)
        mask = np.concatenate([zeros[:, :, np.newaxis], zeros[:, :, np.newaxis], mask_channel[:, :, np.newaxis]], axis=2)
        cv.imwrite(os.path.join(save_dir, vid, one[:-4] + '.png'), mask)

