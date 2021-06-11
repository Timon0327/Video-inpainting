'''
#  filename: youtube.py
#  customize dataset for Yotube-VOS
#  altogether 3 types of dataset for 5 occasions:
#  FlowNet
#  4. generate ground truth flow
#       get: 2 consecutive frames
#       usage: input for inferring stage of FlowNet2
#
#  ResNet
#  5. generate features of corrupted frames
#       get: 2N consecutive frames
#       usage: input for Resnet
#
#  FlownetCG
#  6. training, validation, and testing for FlownetCG
#       get: 2 consecutive corrupted frames, features for neighboring frames (masks for frames, ground truth)
#            (output file names)
#  Likun Qin, 2021
'''
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import cv2 as cv
import numpy as np
import os
import pickle
import cvbase as cvb
from utils.image import apply_mask_resize
from cfgs import config

totensor = ToTensor()


class FlownetInfer(Dataset):
    '''
    dataset for FlowNet2
    '''
    def __init__(self, data_root, out_dir=None, mask_dir=None):
        '''
        initialization
        data_root structure:
            |-- dataroot
                |-- Annotations
                |-- JPEGImages
                |-- masks
                |-- feature (possibly)
                |-- flow (possibly)
                |-- gt (possibly)
        :param data_root: the directory of dataroot
        :param out_dir: the directory to store output flow files
        :param mask_dir: the directory storing masks. If none, use self-generated masks then
        '''
        self.data_root = data_root

        if not out_dir:
            self.out_dir = os.path.join(data_root, 'gt')
        else:
            self.out_dir = out_dir

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # set path
        self.img_dir = os.path.join(data_root, 'JPEGImages')
        if not mask_dir:
            self.mask_dir = os.path.join(data_root, 'masks')
        else:
            self.mask_dir = mask_dir

        # record video names
        self.video_list = os.listdir(self.img_dir)
        try:
            t = self.video_list.index('.DS_Store')
            del self.video_list[t]
        except ValueError:
            pass
        self.video_list.sort()

        self.file = os.path.join(data_root, 'ImageSets', '2017', 'gt.txt')

        if not os.path.exists(self.file):
            with open(self.file, 'w') as f:
                for video in self.video_list:
                    if not os.path.exists(os.path.join(self.out_dir, video)):
                        os.mkdir(os.path.join(self.out_dir, video))
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
                            f.write(os.path.join(os.path.join(self.out_dir, video), imgs[i][:-4] + '.flo'))
                            f.write('\n')

                        if i - 1 >= 0:
                            f.write(os.path.join(os.path.join(self.img_dir, video), imgs[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.img_dir, video), imgs[i - 1]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.out_dir, video), imgs[i][:-4] + '.rflo'))
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

        frame1 = cv.resize(frame1, config.IMG_SIZE)
        frame2 = cv.resize(frame2, config.IMG_SIZE)
        img1 = totensor(frame1)
        img2 = totensor(frame2)

        result = {'frame1': img1,
                  'frame2': img2,
                  'flow_file': self.flow_list[idx]}

        return result


class FlownetCGData(Dataset):
    '''
    dataset for FlowNet2
    '''
    def __init__(self, data_root, mode, out_dir=None, mask_dir=None, feature_dir=None, gt_dir=None):
        '''
        initialization
        data_root structure:
            |-- dataroot
                |-- Annotations
                |-- JPEGImages
                |-- masks
                |-- feature (possibly)
                |-- flow (possibly)
                |-- gt (possibly)
        modes:
            train -- for training, get 2 frames, features, masks of 2 frames, and gt forward flow
            valid -- for validation, get 2 frames, features, masks of 2 frames, and gt forward flow
            test -- for testing, get 2 frames, features, and output filename
        :param data_root: the directory of dataroot
        :param mode: str, 'train' or 'test'
        :param out_dir: the directory to store output flow files
        :param mask_dir: the directory storing masks. If none, use self-generated masks then
        :param feature_dir: the directory storing features. If none, use self-generated features then
        :param gt_dir: the directory storing gt. If none, use self-generated gts then
        '''
        self.data_root = data_root
        self.mode = mode

        if not out_dir:
            self.out_dir = os.path.join(data_root, 'flow')
        else:
            self.out_dir = out_dir

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # set path
        self.img_dir = os.path.join(data_root, 'JPEGImages')  # Full-Resolution
        if not mask_dir:
            self.mask_dir = os.path.join(data_root, 'masks')
        else:
            self.mask_dir = mask_dir
        assert os.path.exists(self.mask_dir)

        if not feature_dir:
            self.feature_dir = os.path.join(data_root, 'feature')
        else:
            self.feature_dir = feature_dir
        assert os.path.exists(self.feature_dir)

        if not gt_dir:
            self.gt_dir = os.path.join(data_root, 'gt')
        else:
            self.gt_dir = gt_dir
        assert os.path.exists(self.gt_dir)

        # if mode == 'train':
        #     self.video_file = os.path.join(data_root, 'ImageSets', '2017', 'train.txt')
        #     self.file = os.path.join(data_root, 'ImageSets', '2017', 'train_gflownet.txt')
        # elif mode == 'valid':
        #     self.video_file = os.path.join(data_root, 'ImageSets', '2017', 'val.txt')
        #     self.file = os.path.join(data_root, 'ImageSets', '2017', 'valid_gflownet.txt')
        # else:
        #     self.video_file = os.path.join(data_root, 'ImageSets', '2017', 'test.txt')
        #     self.file = os.path.join(data_root, 'ImageSets', '2017', 'test_gflownet.txt')
        #
        # with open(self.video_file, 'r') as f:
        #     tmp = f.readlines()
        #     self.video_list = [x[:-1] for x in tmp]
        #     # self.video_list.sort()

        # load video name
        self.video_list = os.listdir(self.img_dir)
        try:
            t = self.video_list.index('.DS_Store')
            del self.video_list[t]
        except ValueError:
            pass
        self.video_list.sort()

        if self.mode == 'train':
            self.file = os.path.join(data_root, 'train_gflownet.txt')
        elif self.mode == 'valid':
            self.file = os.path.join(data_root, 'valid_gflownet.txt')
        else:
            self.file = os.path.join(data_root, 'test_gflownet.txt')

        if not os.path.exists(self.file):
            with open(self.file, 'w') as f:
                for video in self.video_list:
                    if not os.path.exists(os.path.join(self.out_dir, video)):
                        os.mkdir(os.path.join(self.out_dir, video))
                    # get the number of images per video
                    features = os.listdir(os.path.join(self.feature_dir, video))
                    features.sort()
                    feature_num = len(features)

                    images = os.listdir(os.path.join(self.img_dir, video))
                    images.sort()

                    # generate text file
                    for i in range(feature_num):
                        first = features[i].split('.')[0]
                        id = images.index(first + '.jpg')
                        second = images[id + 1].split('.')[0]
                        # file format: frame1 frame2 feature (mask1 mask2 gt) (output)
                        if 'rpk' in features[i]:
                            f.write(os.path.join(os.path.join(self.img_dir, video), second + '.jpg'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.img_dir, video), first + '.jpg'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.feature_dir, video), second + '.rpk'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.mask_dir, video), second + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.mask_dir, video), first + '.png'))
                            if mode == 'train' or mode == 'valid':
                                f.write(' ')
                                f.write(os.path.join(os.path.join(self.gt_dir, video), second + '.rflo'))
                            else:
                                f.write(' ')
                                f.write(os.path.join(os.path.join(self.out_dir, video), second + '.rflo'))
                            f.write('\n')
                        else:

                            f.write(os.path.join(os.path.join(self.img_dir, video), first + '.jpg'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.img_dir, video), second + '.jpg'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.feature_dir, video), features[i]))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.mask_dir, video), first + '.png'))
                            f.write(' ')
                            f.write(os.path.join(os.path.join(self.mask_dir, video), second + '.png'))
                            if mode == 'train' or mode == 'valid':
                                f.write(' ')
                                f.write(os.path.join(os.path.join(self.gt_dir, video), first + '.flo'))
                            else:
                                f.write(' ')
                                f.write(os.path.join(os.path.join(self.out_dir, video), first + '.flo'))
                            f.write('\n')

        self.frame_list1 = []
        self.frame_list2 = []
        self.feature_list = []
        self.mask_list1 = []
        self.mask_list2 = []
        self.gt_list = []
        self.out_list = []

        with open(self.file, 'r') as f:
            for line in f:
                line = line.strip(' ')
                line = line.strip('\n')
                filenames = line.split(' ')
                self.frame_list1.append(filenames[0])
                self.frame_list2.append(filenames[1])
                self.feature_list.append(filenames[2])
                self.mask_list1.append(filenames[3])
                self.mask_list2.append(filenames[4])
                if mode == 'train' or mode == 'valid':
                    self.gt_list.append(filenames[5])
                else:
                    self.out_list.append(filenames[-1])

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

        mask1 = cv.imread(self.mask_list1[idx])[:, :, 2]
        mask2 = cv.imread(self.mask_list2[idx])[:, :, 2]

        img1 = apply_mask_resize(frame1, size=config.IMG_SIZE, slice=0, mask=mask1)
        img2 = apply_mask_resize(frame2, size=config.IMG_SIZE, slice=0, mask=mask2)

        imgs = torch.cat([img1, img2], dim=0)

        with open(os.path.join(self.feature_dir, self.feature_list[idx]), 'rb') as f:
            feature = pickle.load(f)
            feature = torch.cat(feature, dim=0)
            # feature.requires_grad_(False)

        if self.mode != 'test':
            gt = cvb.read_flow(os.path.join(self.gt_dir, self.gt_list[idx]))
            gt = torch.from_numpy(gt[:, :, :]).permute(2, 0, 1)
            gt.requires_grad_(False)
            result = {
                'frames': imgs,
                'feature': feature,
                'mask1': mask1,
                'mask2': mask2,
                'gt': gt
            }
        else:
            result = {'frames': imgs,
                      'feature': feature,
                      'out_file': self.out_list[idx]}

        return result


class ResnetInfer(Dataset):
    '''
    dataset for flow extraction
    '''
    def __init__(self, data_root, mask_dir, out_dir, slice, N):
        '''
        initialize dataset
        data_root structure:
            |-- dataroot
                |-- Annotations
                |-- JPEGImages
                |-- masks
                |-- feature (possibly)
                |-- flow (possibly)
                |-- gt (possibly)
        :param data_root: the directory of data_root
        :param mask_dir: the directory of masks. if None, then use self-generated masks
        :param out_dir: the directory to store features
        :param slice: how many patches each frame is divided into in each direction, int
        :param div: how many batch are needed to process one frame (to save CUDA memory), int  (abandoned)
        :param N: get 2N frames in total, N frames before, N frames after. int

        notice: the number of nodes in GCN is actually (2N) * slice * slice
        '''
        self.slice = slice
        self.N = N
        all_intervals = [0, 1, -1, 2, -2, 3, -4, 5, -7, 8]

        # select frames with different offsets to form the nodes
        self.intervals = all_intervals[:2 * N]
        self.intervals.sort()

        self.data_root = data_root
        self.img_dir = os.path.join(data_root, 'JPEGImages')

        if mask_dir:
            self.mask_dir = mask_dir
        else:
            self.mask_dir = os.path.join(data_root, 'masks')
        assert os.path.exists(self.mask_dir)

        if out_dir:
            self.out_dir = out_dir
        else:
            self.out_dir = os.path.join(data_root, 'feature')

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # record video names
        self.video_list = os.listdir(self.img_dir)
        try:
            t = self.video_list.index('.DS_Store')
            del self.video_list[t]
        except ValueError:
            pass
        self.video_list.sort()

        self.file = os.path.join(data_root, 'feature.txt')

        if not os.path.exists(self.file):
            with open(self.file, 'w') as f:

                # margin = self.intervals[N-1]
                # offsets = [0]
                # for j in range(N):
                #     offsets.append(self.intervals[j])
                #     offsets.append(-self.intervals[j])
                # offsets.sort()

                margin_left = self.intervals[0]
                margin_right = self.intervals[-1]

                for video in self.video_list:
                    if not os.path.exists(os.path.join(self.out_dir, video)):
                        os.mkdir(os.path.join(self.out_dir, video))
                    # get the number of images per video
                    imgs = os.listdir(os.path.join(self.img_dir, video))
                    imgs.sort()
                    img_num = len(imgs)

                    # generate text file
                    for i in range(img_num):
                        if i + margin_right < img_num and i + margin_left >= 0:
                            for offset in self.intervals:
                                f.write(os.path.join(video, imgs[i + offset]))
                                f.write(' ')
                            f.write(os.path.join(video, imgs[i][:-4] + '.pk'))
                            f.write('\n')
                            for offset in self.intervals[::-1]:
                                f.write(os.path.join(video, imgs[i + offset]))
                                f.write(' ')
                            f.write(os.path.join(video, imgs[i + 1][:-4] + '.rpk'))
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
                for k in range(2 * N):
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

            img = apply_mask_resize(frame, size=config.IMG_SIZE, slice=self.slice, mask=mask)
            frames.append(img)

        result = {'frames': frames,
                  'out_file': os.path.join(self.out_dir, self.out_files[idx])}

        return result


class GFCNetData(Dataset):
    '''
    dataset for Graph Flow Completion Network
    '''
    def __init__(self, data_root, mode, flow_dir=None, feature_dir=None, mask_dir=None, gt_dir=None):
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
        :param mode: 'train', 'val' or 'test'
        :param flow_dir: the directory of flow files. If None, then use the one in data_root directory.
        :param feature_dir: the directory of feature files. If None, then use the one in data_root directory.
        :param mask_dir: the directory of mask files. If None, then use the one in data_root directory.
        :param gt_dir: the directory of gt flow files. If None, then use the one in data_root directory.
        '''
        self.data_root = data_root
        self.mode = mode

        # set path
        self.mask_dir = mask_dir if not mask_dir else os.path.join(data_root, 'masks')
        self.flow_dir = flow_dir if not flow_dir else os.path.join(data_root, 'flow')
        self.feature_dir = feature_dir if not feature_dir else os.path.join(data_root, 'feature')
        self.gt_dir = gt_dir if not gt_dir else os.path.join(data_root, 'gt')

        # load video names
        if mode == 'train':
            self.video_file = os.path.join(data_root, 'ImageSets', '2017', 'train.txt')
        elif mode == 'val':
            self.video_file = os.path.join(data_root, 'ImageSets', '2017', 'val.txt')
        else:
            self.video_file = os.path.join(data_root, 'ImageSets', '2017', 'test.txt')

        with open(self.video_file, 'r') as f:
            tmp = f.readlines()
            self.video_list = [x[:-1] for x in tmp]
            self.video_list.sort()

        self.feature_list = []
        self.flow_list_f = []          # list for forward flow
        self.flow_list_b = []          # list for backward flow
        self.mask1 = []                # list for previous mask
        self.mask2 = []                # list for following mask
        self.gt_list_f = []            # list for ground truth of forward flow
        self.gt_list_b = []            # list for ground truth of backward flow

        for video in self.video_list:
            feature_names = os.listdir(os.path.join(self.feature_dir, video))
            feature_names.sort()

            # get the list of all backward flow files
            rflo_names = []
            for file in os.listdir(os.path.join(self.flow_dir, video)):
                if file.endswith('.rflo'):
                    rflo_names.append(file)
            rflo_names.sort()

            for one in feature_names:
                img_id = one.split('.')[0]
                rflo_id = rflo_names.index(img_id + '.rflo') + 1
                img2_id = rflo_names[rflo_id].split('.')[0]

                self.feature_list.append(os.path.join(video, one))

                self.flow_list_f.append(os.path.join(video, img_id + '.flo'))
                self.flow_list_b.append(os.path.join(video, rflo_names[rflo_id]))

                self.mask1.append(os.path.join(video, img_id + '.png'))
                self.mask2.append(os.path.join(video, img2_id + '.png'))

                self.gt_list_f.append(os.path.join(video, img_id + '.flo'))
                self.gt_list_b.append(os.path.join(video, img2_id + '.rflo'))

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, idx):
        '''
        get corrupted forward and backward flow, features of 2N frames
        if the mode is 'train' or 'val, get 2 masks and 2 ground-truth flow in addition

        :param idx:
        :return:
        '''

        with open(os.path.join(self.feature_dir, self.feature_list[idx]), 'rb') as f:
            feature = pickle.load(f)
            feature = torch.from_numpy(feature)
            feature.requires_grad_(False)

        flow_f = cvb.read_flow(os.path.join(self.flow_dir, self.flow_list_f[idx]))
        flow_b = cvb.read_flow(os.path.join(self.flow_dir, self.flow_list_b[idx]))
        flow_f = torch.from_numpy(flow_f[np.newaxis, :, :, :])
        flow_b = torch.from_numpy(flow_b[np.newaxis, :, :, :])
        flows = torch.cat([flow_f, flow_b], dim=0)
        flows.requires_grad_(False)

        if self.mode == 'train' or self.mode == 'val':
            mask1 = cv.imread(os.path.join(self.mask_dir, self.mask1[idx]))[:, :, 2]
            mask2 = cv.imread(os.path.join(self.mask_dir, self.mask2[idx]))[:, :, 2]

            mask1 = np.concatenate([mask1[:, :, np.newaxis], mask1[:, :, np.newaxis]], axis=2)
            mask2 = np.concatenate([mask2[:, :, np.newaxis], mask2[:, :, np.newaxis]], axis=2)

            mask1 = torch.from_numpy(mask1[np.newaxis, :, :, :])
            mask2 = torch.from_numpy(mask2[np.newaxis, :, :, :])
            masks = torch.cat([mask1, mask2], dim=0)
            masks.requires_grad_(False)

            gt_f = cvb.read_flow(os.path.join(self.gt_dir, self.gt_list_f[idx]))
            gt_b = cvb.read_flow(os.path.join(self.gt_dir, self.gt_list_b[idx]))

            gt_f = torch.from_numpy(gt_f[np.newaxis, :, :, :])
            gt_b = torch.from_numpy(gt_b[np.newaxis, :, :, :])
            gts = torch.cat([gt_f, gt_b], dim=0)
            gts.requires_grad_(False)

            result = {'feature': feature,       # shape [nodes, 2048]
                      'flows': flows,           # shape [2, height, width, 2]
                      'masks': masks,           # shape [2, height, width, 2]
                      'gts': gts                # shape [2, height, width, 2]
                     }

            return result
        else:
            result = {'feature': feature,
                      'flows': flows,
                      'output_forward': self.flow_list_f[idx],
                      'output_backward': self.flow_list_b[idx]}
            return result


if __name__ == '__main__':
    # dataset = FlownetInfer(data_root='/home/captain/dataset/tiny_DAVIS',
    #                         mode='restore',
    #                         out_dir='/home/captain/dataset/tiny_DAVIS/flow',
    #                         mask_dir=None)
    # dataset = ResnetInfer(data_root='/home/captain/dataset/tiny_DAVIS',
    #                       mask_dir=None,
    #                       out_dir=None,
    #                       slice=config.SLICE,
    #                       N=config.N)
    dataset = FlownetCGData(data_root='/home/cap/dataset/tiny_DAVIS', mode='valid')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # res = dataset.__getitem__(5)
    for i, result in enumerate(dataloader):
        if i == 2:
            break
        print(i)
        print(result)
    print('data loaded')











