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
import cvbase as cvb

class Davis_dataset(Dataset):
    def __init__(self,
                 data_root,
                 flow_dir,
                 gt_dir,
                 size,
                 mask_dir=None,
                 train_file='./video_train.txt',
                 test_file='./video_test.txt',
                 val_file='./video_val.txt',
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

        there are 4 modes:
        train, val -- get 2 adjacent frames, 2 mask of frames, 1 flow file, 1 gt flow file
                        training or validation for GCN and flow completion
        test -- get 2 adjacent frames, 2 mask of frames, 1 flow file
                test for GCN and flow completion
        flow -- get 2 adjacent frames, 2 mask of frames, 1 flow file path
                generate flow of masked frames
        :param data_root: the directory containing Annotations, JPEGImages, ImageSets
        :param flow_dir: the directory to save flow files
        :param gt_dir: the directory to save ground -ruth flow files, has the save stucture as JPEGImages
        :param size: the size of image, tuple, (height, width)
        :param slice: how many slices in each dimension (height, width), int
        :param mask_dir: the directory saving masks
        :param train_file: (for train)the path of the text file which stores filenames of the frames, the flow, and gt
        :param test_file: (for test)the path of the text file which stores filenames of the frames, the flow, and gt
        :param val_file: (for validation) ...
        :param div: the number of batches one image to be divided, int
        :param transform: the function to be applied to images
        :param mode: str  train, val , test or flow
        '''

        # set up where flow files are stored
        self.flow_dir = flow_dir
        if not os.path.exists(flow_dir):
            os.mkdir(flow_dir)

        # set up ground-truth directory
        self.gt_dir = gt_dir

        # set up mask directory
        self.mask_dir = mask_dir

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
        elif mode == 'test':
            self.images_dir = os.path.join(data_root, 'JPEGImages', 'Full-Resolution')
            video_file = os.path.join(data_root, 'ImageSets', '2017', 'test-dev.txt')
        else:
            self.div = 1
            self.slice = 1

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

        # check if text file is ready
        if mode == 'train':
            the_file = train_file
        elif mode == 'test':
            the_file = test_file
        elif mode == 'val':
            the_file = val_file
        else:
            the_file = './flow.txt'

        if mode in ['train', 'val'] and not os.path.exists(the_file):
            with open(the_file, 'w') as f:
                for one in self.video_list:
                    # get the number of images per video
                    imgs = os.listdir(os.path.join(self.images_dir, one))
                    img_num = len(imgs)

                    # generate text file
                    for i in range(img_num):
                        if i + 1 < img_num:
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i + 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(mask_dir, 'one'), imgs[i][:-4] + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(mask_dir, 'one'), imgs[i + 1][:-4] + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(flow_dir, 'one'), imgs[i][:-4] + '.flo'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(gt_dir, 'one'), imgs[i][:-4] + '.flo'))
                            f.write('\n')

                        if i - 1 >= 0:
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i - 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(mask_dir, 'one'), imgs[i][:-4] + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(mask_dir, 'one'), imgs[i - 1][:-4] + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(flow_dir, 'one'), imgs[i][:-4] + '.rflo'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(gt_dir, 'one'), imgs[i][:-4] + '.rflo'))
                            f.write('\n')

                    # record the number of image in each video
                    self.frame_num.append(img_num)
                    acc += img_num
                    self.acc_num.append(acc)
                self.total = np.sum(self.frame_num)

        elif mode == 'test' and not os.path.exists(the_file):
            with open(the_file, 'w') as f:
                for one in self.video_list:
                    # get the number of images per video
                    imgs = os.listdir(os.path.join(self.images_dir, one))
                    img_num = len(imgs)

                    # make directory for flow files
                    os.mkdir(os.path.join(flow_dir, 'one'))

                    # generate text file
                    for i in range(img_num):
                        if i + 1 < img_num:
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i + 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(flow_dir, 'one'), imgs[i][:-4] + '.flo'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.annotations_dir, 'one'), imgs[i][:-4] + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.annotations_dir, 'one'), imgs[i + 1][:-4] + '.png'))
                            f.write('\n')

                        if i - 1 >= 0:
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i - 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(flow_dir, 'one'), imgs[i][:-4] + '.rflo'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.annotations_dir, 'one'), imgs[i][:-4] + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.annotations_dir, 'one'), imgs[i - 1][:-4] + '.png'))
                            f.write('\n')

                    # record the number of image in each video
                    self.frame_num.append(img_num)
                    acc += img_num
                    self.acc_num.append(acc)
                self.total = np.sum(self.frame_num)

        elif mode in ['train', 'val', 'test']:
            # no need to generate text file
            for one in self.video_list:
                imgs = os.listdir(os.path.join(self.images_dir, one))
                img_num = len(imgs)
                self.frame_num.append(img_num)
                acc += img_num
                self.acc_num.append(acc)
            self.total = np.sum(self.frame_num)
        elif mode is 'flow':
            with open(the_file, 'w') as f:
                for one in self.video_list:
                    # get the number of images per video
                    imgs = os.listdir(os.path.join(self.images_dir, one))
                    img_num = len(imgs)

                    # make directory for flow files
                    os.mkdir(os.path.join(flow_dir, 'one'))

                    # generate text file
                    for i in range(img_num):
                        if i + 1 < img_num:
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i + 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(flow_dir, 'one'), imgs[i][:-4] + '.flo'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.annotations_dir, 'one'), imgs[i][:-4] + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.annotations_dir, 'one'), imgs[i + 1][:-4] + '.png'))
                            f.write('\n')

                        if i - 1 >= 0:
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.images_dir, one), imgs[i - 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(flow_dir, 'one'), imgs[i][:-4] + '.rflo'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.annotations_dir, 'one'), imgs[i][:-4] + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.annotations_dir, 'one'), imgs[i - 1][:-4] + '.png'))
                            f.write('\n')
                    self.frame_num.append(img_num)
                    acc += img_num
                    self.acc_num.append(acc)
                self.total = np.sum(self.frame_num)

        self.frame1_list = []
        self.frame2_list = []
        self.flow_list = []
        self.gt_list = []
        self.mask1_list = []
        self.mask2_list = []
        self.length = 0

        with open(the_file, 'r') as f:
            for line in f:
                self.length += 1
                line = line.strip(' ')
                line = line.strip('\n')

                line_split = line.split(' ')
                if mode in ['train', 'val']:
                    self.frame1_list.append(line_split[0])
                    self.frame2_list.append(line_split[1])
                    self.mask1_list.append(line_split[2])
                    self.mask2_list.append(line_split[3])
                    self.flow_list.append(line_split[4])
                    self.gt_list.append(line_split[5])
                else:
                    self.frame1_list.append(line_split[0])
                    self.frame2_list.append(line_split[1])
                    self.flow_list.append(line_split[2])
                    self.mask1_list.append(line_split[3])
                    self.mask2_list.append(line_split[4])

    def __len__(self):
        '''

        :return:
        '''
        return self.length

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
        # imgs_list = os.listdir(os.path.join(self.images_dir, self.video_name))
        # imgs_list.sort()
        # frame = cv.imread(os.path.join(self.images_dir, self.video_name, imgs_list[img_id]))

        frame1 = cv.imread(self.frame1_list[idx])
        frame2 = cv.imread(self.frame2_list[idx])

        mask1 = cv.imread(self.frame1_list[idx])[2]
        mask2 = cv.imread(self.frame2_list[idx])[2]

        whole_frames1, mask1 = self.transform(frame=frame1,
                                            size=self.size,
                                            slice=self.slice,
                                            mask=mask1,
                                            margin=(50,50),
                                            rect_shape=(270,480))
        whole_frames2, mask2 = self.transform(frame=frame2,
                                            size=self.size,
                                            slice=self.slice,
                                            mask=mask2,
                                            margin=(50, 50),
                                            rect_shape=(270, 480))   # whole_frames1 and 2 can be a Tensor in [C, H, W]

        # divide one image to several batches
        frames1 = []
        if self.div != 1:
            for one in torch.chunk(whole_frames1, self.div, dim=0):
                frames1.append(one)
        else:
            frames1.append(whole_frames1)

        frames2 = []
        if self.div != 1:
            for one in torch.chunk(whole_frames2, self.div, dim=0):
                frames2.append(one)
        else:
            frames2.append(whole_frames2)

        if self.mode in ['train', 'val']:
            flow = cvb.read_flow(self.flow_list[idx])
            gt = cvb.read_flow(self.gt_list[idx])
            result = {'image1': frames1,
                      'image2': frames2,
                      'mask1': mask1,
                      'mask2': mask2,
                      'flow': flow,
                      'gt': gt,
                      'video': self.video_name,
                      'id': img_id,
                      }
        elif self.mode is 'test':
            flow = cvb.read_flow(self.flow_list[idx])
            result = {'image1': frames1,
                      'image2': frames2,
                      'mask1': mask1,
                      'mask2': mask2,
                      'flow': flow,
                      'video': self.video_name,
                      'id': img_id,
                      }
        else:
            result = {'image1': frames1,
                      'image2': frames2,
                      'mask1': mask1,
                      'mask2': mask2,
                      'flow_path': self.flow_list[idx],
                      'video': self.video_name,
                      'id': img_id,
                      }

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











