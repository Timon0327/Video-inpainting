'''
#  filename: davis.py
#  customize dataset for DAVIS
#  Likun Qin, 2021
'''
import torch
from torch.utils.data.dataset import Dataset
from utils.image import image_and_mask
import cv2 as cv
import numpy as np
import os

class Davis_dataset(Dataset):
    def __init__(self,
                 data_root,
                 size,
                 slice=10,
                 div=1,
                 mode='val',
                 transform=None):
        '''
        initialization of Dasvis_dataset
        dataset structure:
        data_root | -- Annotations
                  | -- JPEGImages
                  | -- ImageSets
        :param data_root: the directory containing Annotations, JPEGImages, ImageSets
        :param size: the size of image, tuple, (height, width)
        :param slice: how many slices in each dimension (height, width), int
        :param div: the number of batches one image to be divided, int
        :param transform: the function to be applied to images
        :param mode: str   train, val or test
        '''
        self.size = size
        self.slice = slice
        self.div = div
        # set path
        if mode == 'train':
            self.images_dir = os.path.join(data_root, 'JPEGImages', 'Full-Resolution')   # Full-Resolution
            self.annotations_dir = os.path.join(data_root, 'Annotations', 'Full-Resolution')
            video_file = os.path.join(data_root, 'ImageSets', '2017', 'train.txt')
        elif mode == 'val':
            self.images_dir = os.path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.annotations_dir = os.path.join(data_root, 'Annotations', 'Full-Resolution')
            video_file = os.path.join(data_root, 'ImageSets', '2017', 'val.txt')
        else:
            self.images_dir = os.path.join(data_root, 'JPEGImages', 'Full-Resolution')
            video_file = os.path.join(data_root, 'ImageSets', '2017', 'test-dev.txt')

        # set functions to adjust images
        self.transform = transform

        # set mode
        self.mode = mode

        # read in names of video to be loaded
        self.video_list = None
        with open(video_file, 'r') as f:
            tmp = f.readlines()
            self.video_list = [x[:-1] for x in tmp]

        # video name
        self.video_name = None

        # counting number
        self.count = 0

        # total number of images
        self.total = 0

        # number of frames of each video
        self.frame_num = []

        # accumulated number of all images
        self.acc_num = [0]
        acc = 0

        for one in self.video_list:
            imgs = os.listdir(os.path.join(self.images_dir, one))
            self.frame_num.append(len(imgs))
            acc += len(imgs)
            self.acc_num.append(acc)

        self.total = np.sum(self.frame_num)

    def __len__(self):
        '''

        :return:
        '''
        return self.total

    def __getitem__(self, idx):
        '''
        get one image in the dataset
        :param idx: the id of the image, int
        :return:
        '''
        # find the video the image corresponds to
        video_id = 0
        for i, acc in enumerate(self.acc_num):
            if idx >= acc:
                continue
            else:
                video_id = i - 1
                break
        self.video_name = self.video_list[video_id]

        # find and load the frame
        img_id = idx - self.acc_num[video_id]
        imgs_list = os.listdir(os.path.join(self.images_dir, self.video_name))
        imgs_list.sort()
        frame = cv.imread(os.path.join(self.images_dir, self.video_name, imgs_list[img_id]))

        if self.mode == 'test':
            mask = cv.imread(os.path.join(self.annotations_dir, self.video_name, imgs_list[img_id]))[2]
        else:
            mask = None
        whole_frames, mask = self.transform(frame=frame,
                                      size=self.size,
                                      slice=self.slice,
                                      mask=mask,
                                      margin=(50,50),
                                      rect_shape=(270,480))

        # divide one image to several batches
        frames = []
        if self.div != 1:
            for one in torch.chunk(whole_frames, self.div, dim=0):
                frames.append(one)
        else:
            frames.append(whole_frames)

        result = {'image': frames,
                  'video': self.video_name,
                  'id': img_id,
                  'mask': mask}

        return result


if __name__ == '__main__':
    dataset = Davis_dataset(data_root='/home/captain/dataset/DAVIS/DAVIS-semisupervised/DAVIS-trainval',
                            size=(800, 800),
                            slice=4,
                            div=4,
                            mode='val',
                            transform=image_and_mask
                            )
    print(len(dataset))
    res = dataset.__getitem__(100)
    print('data loaded')











