'''
#  filename: config.py
#  hyper-parameters
#  By Likun Qin, 2021
'''

'''
############### General Settings ################
'''
IMG_SIZE = (640, 640)       # (640, 640)

# dataset
DATASET = 'ytb'  # or 'ytb'

# directory of test frames
TEST_ROOT = '/mnt/qinlikun/dataset/test/ytb'

# directory of valid frames
VALID_ROOT = '/mnt/qinlikun/dataset/youtube/valid'

# corrupted area has size of (MASK_RATIO * height, MASK_RATIO * width)
MASK_RATIO = 0.25

# mask type, 'random' is random bbox, 'mid' is mid bbox
MASK_TYPE = 'mid'

# each image is divided by SLICE * SLICE patches
SLICE = 2

# how many batch are needed to process one frame (to save CUDA memory)
# DIV = 1

# how many frames, before and after separately, are used to build temporal GCN
N = 4

# FlowNet pretrained weight
FLOWNET_WEIGHT = '/mnt/qinlikun/inpainting/flownet/FlowNet2_checkpoint.pth.tar'

# Resnet pretrained weight
RESNET50_WEIGHT = '/mnt/qinlikun/inpainting/resnet/resnet50-19c8e357.pth'
RESNET101_WEIGHT = '/mnt/qinlikun/inpainting/resnet/resnet101-5d3b4d8f.pth '

# image inpainting model
DEEPFILL_WEIGHT = '/mnt/qinlikun/inpainting/imagenet_deepfill.pth'

'''
############### Training Setting ################
'''

DATA_ROOT = '/mnt/qinlikun/dataset/youtube/train'

SAVE_DIR = '/mnt/qinlikun/inpainting'

# maximum steps
MAX_ITER = 21000    # 21000
DECAY_EPOCHES = [5, 8]  # [5, 8]

EPOCH = 11   # 11

BATCH_SIZE = 36

# number of workers for dataloader
N_THREADS = 8

# learning rate
LR = 1e-5   # 1e-4

WEIGHTED_DECAY = 0.00004

LAMBDA_SMOOTH = 0.1
LAMBDA_HARD = 2.



'''
############### Network Setting ################
'''
# correlation layer
CORR_PAD_SIZE = 20
CORR_KERNEL_SIZE = 1
CORR_MAX_DISPLACEMENT = 20
CORR_STRIDE1 = 1
CORR_STRIDE2 = 2
CORR_MULTIPLY = 1

