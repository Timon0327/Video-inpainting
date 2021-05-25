'''
#  filename: config.py
#  hyper-parameters
#  By Likun Qin, 2021
'''

'''
############### General Settings ################
'''
IMG_SIZE = (800, 800)       # (1792, 1792)
DATA_ROOT = '/mnt/qinlikun/dataset/tiny_DAVIS'

# corrupted area has size of (MASK_RATIO * height, MASK_RATIO * width)
MASK_RATIO = 0.25

# each image is divided by SLICE * SLICE patches
SLICE = 2

# how many batch are needed to process one frame (to save CUDA memory)
DIV = 1

# how many frames, before and after separately, are used to build temporal GCN
N = 4

# FlowNet pretrained weight
FLOWNET_WEIGHT = '/mnt/qinlikun/inpainting/flownet/FlowNet2_checkpoint.pth.tar'

# Resnet pretrained weight
RESNET50_WEIGHT = '/mnt/qinlikun/inpainting/resnet/resnet50-19c8e357.pth'
RESNET101_WEIGHT = '/mnt/qinlikun/inpainting/resnet/resnet101-5d3b4d8f.pth '

'''
############### Training Setting ################
'''

SAVE_DIR = '/mnt/qinlikun/inpainting/ckpt'

# maximum steps
MAX_ITER = 100000

BATCH_SIZE = 1

# number of workers for dataloader
N_THREADS = 8

# learning rate
LR = 1e-4

WEIGHTED_DECAY = 0.00004

LAMBDA_SMOOTH = 0.1
LAMBDA_HARD = 2.



