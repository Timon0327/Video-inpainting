'''
#  filename: davis.py
#  customize dataset for DAVIS
#  altogether 3 types of dataset for 5 occasions:
#  GFCNet
#  1. for training and validation
#       get: corrupted flows (forward and backward), features of 2N+1 neighboring frames, mask, ground-truth flows
#       usage: training GCN and flow completion network in the back
#  2. for test:
#       get: corrupted flows (forward and backward), features of 2N+1 neighboring frames
#       usage: test GCN and flow completion network in the back
#
#  FlowNet
#  3. generate corrupted flow
#       get: 2 consecutive frames
#       usage: input for inferring stage of FlowNet2
#  4. generate ground truth flow
#       get: 2 consecutive frames
#       usage: input for inferring stage of FlowNet2
#
#  ResNet
#  5. generate features of corrupted frames
#       get: 2N+1 consecutive frames
#       usage: input for Resnet
#
#  Likun Qin, 2021
'''
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import cv2 as cv
import numpy as np
import os
import cvbase as cvb
from utils.image import apply_mask_resize
from cfgs import config

totensor = ToTensor()


class FlownetInfer(Dataset):
    '''
    dataset for FlowNet2
    '''
    def __init__(self, data_root, mode, out_dir, mask_dir=None):
        '''
        initialization
        data_root structure:
            |-- dataroot
                |-- Annotations
                |-- JPEGImages
                |-- masks
                |-- ImageSets
                |-- feature (possibly)
                |-- flow (possibly)
                |-- gt (possibly)
        modes:
            gt -- generate ground-truth with raw complete frames
            restore -- generate flow with corrupted frames
        :param data_root: the directory of dataroot
        :param mode: str, 'gt' or 'restore'
        :param out_dir: the directory to store output flow files
        :param mask_dir: the directory storing masks. If none, use self-generated masks then
        '''
        self.data_root = data_root
        self.mode = mode
        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # set path
        self.img_dir = os.path.join(data_root, 'JPEGImages', '480p')  # Full-Resolution
        if not mask_dir:
            self.mask_dir = os.path.join(data_root, 'masks')
        else:
            self.mask_dir = mask_dir

        # record video names
        self.video_list = os.listdir(self.img_dir)
        self.video_list.sort()

        if mode == 'gt':
            self.file = os.path.join(data_root, 'ImageSets', '2017', 'gt.txt')
        else:
            self.file = os.path.join(data_root, 'ImageSets', '2017', 'bad_flow.txt')

        if not os.path.exists(self.file):
            with open(self.file, 'w') as f:
                for video in self.video_list:
                    os.mkdir(os.path.join(out_dir, video))
                    # get the number of images per video
                    imgs = os.listdir(os.path.join(self.img_dir, video))
                    imgs.sort()
                    img_num = len(imgs)

                    # generate text file
                    for i in range(img_num):
                        if i + 1 < img_num:
                            f.write(os.path.join(os.path.join(self.img_dir, video), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.img_dir, video), imgs[i + 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(out_dir, video), imgs[i][:-4] + '.flo'))
                            if mode == 'restore':
                                f.write(' ')
                                f.write(os.path.join(os.path.join(self.mask_dir, video), imgs[i][:-4] + '.png'))
                                f.write(' ')
                                f.write(os.path.join(os.path.join(self.mask_dir, video), imgs[i + 1][:-4] + '.png'))
                            f.write('\n')

                        if i - 1 >= 0:
                            f.write(os.path.join(os.path.join(self.img_dir, video), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.img_dir, video), imgs[i - 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(out_dir, video), imgs[i][:-4] + '.rflo'))
                            if mode == 'restore':
                                f.write(' ')
                                f.write(os.path.join(os.path.join(self.mask_dir, video), imgs[i][:-4] + '.png'))
                                f.write(' ')
                                f.write(os.path.join(os.path.join(self.mask_dir, video), imgs[i - 1][:-4] + '.png'))
                            f.write('\n')


        self.frame_list1 = []
        self.frame_list2 = []
        self.flow_list = []
        self.mask_list1 = []
        self.mask_list2 = []

        with open(self.file, 'r') as f:
            for line in f:
                line = line.strip(' ')
                line = line.strip('\n')
                filenames = line.split(' ')
                self.frame_list1.append(filenames[0])
                self.frame_list2.append(filenames[1])
                self.flow_list.append(filenames[2])
                if mode == 'restore':
                    self.mask_list1.append(filenames[3])
                    self.mask_list2.append(filenames[4])

    def __len__(self):
        return len(self.frame_list1)

    def __getitem__(self, idx):
        '''
        get 2 consecutive frames and the filename of output flow file
        :param idx:
        :return:
        '''

        frame1 = cv.imread(self.frame_list1[idx])
        frame2 = cv.imread(self.frame_list2[idx])

        if self.mode == 'gt':
            frame1 = cv.resize(frame1, config.IMG_SIZE)
            frame2 = cv.resize(frame2, config.IMG_SIZE)
            img1 = totensor(frame1)
            img2 = totensor(frame2)

        else:
            mask1 = cv.imread(self.mask_list1[idx])[:, :, 2]
            mask2 = cv.imread(self.mask_list2[idx])[:, :, 2]

            img1 = apply_mask_resize(frame1, size=config.IMG_SIZE, slice=0, mask=mask1)
            img2 = apply_mask_resize(frame2, size=config.IMG_SIZE, slice=0, mask=mask2)

        result = {'frame1': img1,
                  'frame2': img2,
                  'flow_file': self.flow_list[idx]}

        return result


class ResnetInfer(Dataset):
    '''
    dataset for flow extraction
    '''
    def __init__(self, data_root, mask_dir, out_dir, slice, div, N):
        '''
        initialize dataset
        data_root structure:
            |-- dataroot
                |-- Annotations
                |-- JPEGImages
                |-- masks
                |-- ImageSets
                |-- feature (possibly)
                |-- flow (possibly)
                |-- gt (possibly)
        :param data_root: the directory of data_root
        :param mask_dir: the directory of masks. if None, then use self-generated masks
        :param out_dir: the directory to store features
        :param slice: how many patches each frame is divided into in each direction, int
        :param div: how many batch are needed to process one frame (to save CUDA memory), int
        :param N: get 2N + 1 frames in total, N frames before, N frames after. int

        notice: the number of nodes in GCN is actually (2N+1) * slice * slice
        '''
        self.slice = slice
        self.div = div
        self.N = N
        self.intervals = [1,2,4,6,9,12]

        self.data_root = data_root
        self.img_dir = os.path.join(data_root, 'JPEGImages', '480p')    # Full-Resolution

        if mask_dir:
            self.mask_dir = mask_dir
        else:
            self.mask_dir = os.path.join(data_root, 'masks')

        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # record video names
        self.video_list = os.listdir(self.img_dir)
        self.video_list.sort()

        self.file = os.path.join(data_root, 'ImageSets', '2017', 'feature.txt')

        if not os.path.exists(self.file):
            with open(self.file, 'w') as f:

                margin = self.intervals[N-1]
                offsets = [0]
                for j in range(N):
                    offsets.append(self.intervals[j])
                    offsets.append(-self.intervals[j])
                offsets.sort()

                for video in self.video_list:
                    os.mkdir(os.path.join(out_dir, video))
                    # get the number of images per video
                    imgs = os.listdir(os.path.join(self.img_dir, video))
                    imgs.sort()
                    img_num = len(imgs)

                    # generate text file
                    for i in range(img_num):
                        if i + margin < img_num and i - margin >= 0:
                            for offset in offsets:
                                f.write(os.path.join(video, imgs[i + offset]))
                                f.write(' ')
                            f.write(os.path.join(video, imgs[i][:-4] + '.pk'))
                            f.write('\n')
                        else:
                            continue

        # read in the txt file
        self.frames = []
        self.out_files = []
        with open(self.file, 'r') as file:
            for line in file:
                line = line.strip(' ')
                line = line.strip('\n')
                filenames = line.split(' ')

                tmp_frame = []
                for k in range(2 * N + 1):
                    tmp_frame.append(filenames[k])

                self.frames.append(tmp_frame)
                self.out_files.append(filenames[-1])

    def __len__(self):
        return len(self.out_files)

    def __getitem__(self, idx):
        '''
        get 2N+1 frames in a list, and the name of the output pk file
        :param idx:
        :return:
        '''
        frames = []

        for one in self.frames[idx]:
            frame = cv.imread(os.path.join(self.img_dir, one))
            mask = cv.imread(os.path.join(self.mask_dir, one[:-4] + '.png'))[:, :, 2]

            img = apply_mask_resize(frame, size=config.IMG_SIZE, slice=0, mask=mask)
            frames.append(img)

        result = {'frames': frames,
                  'out_file': os.path.join(self.out_dir, self.out_files[idx])}

        return result


class GFCNetData(Dataset):
    '''
    dataset for Graph Flow Completion Network
    '''
    def __init__(self, data_root, flow_dir=None, feature_dir=None, mask_dir=None, gt_dir=None):
        '''
        initialize dataset for GFCNet
        data_root structure:
            |-- dataroot
                |-- Annotations
                |-- JPEGImages
                |-- masks
                |-- ImageSets
                |-- feature (possibly)
                |-- flow (possibly)
                |-- gt (possibly)
        :param data_root: the directory of data_root
        :param flow_dir: the directory of flow files. If None, then use the one in data_root directory.
        :param feature_dir: the directory of feature files. If None, then use the one in data_root directory.
        :param mask_dir: the directory of mask files. If None, then use the one in data_root directory.
        :param gt_dir: the directory of gt flow files. If None, then use the one in data_root directory.
        '''
        self.data_root = data_root

        self.mask_dir = mask_dir if not mask_dir else os.path.join(data_root, 'masks')
        self.flow_dir = flow_dir if not flow_dir else os.path.join(data_root, 'flow')
        self.feature_dir = feature_dir if not feature_dir else os.path.join(data_root, 'feature')
        self.gt_dir = gt_dir if not gt_dir else os.path.join(data_root, 'gt')

















class Davis_dataset_ref(Dataset):
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
                    imgs.sort()
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
    # dataset = FlownetInfer(data_root='/home/captain/dataset/tiny_DAVIS',
    #                         mode='restore',
    #                         out_dir='/home/captain/dataset/tiny_DAVIS/flow',
    #                         mask_dir=None)
    dataset = ResnetInfer(data_root='/home/captain/dataset/tiny_DAVIS',
                          mask_dir=None,
                          out_dir='/home/captain/dataset/tiny_DAVIS/feature',
                          slice=config.SLICE,
                          div=config.DIV,
                          N=config.N)
    print(len(dataset))
    res = dataset.__getitem__(5)
    print('data loaded')











