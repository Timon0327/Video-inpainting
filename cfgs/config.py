'''
#  filename: config.py
#  hyper-parameters
#  By Likun Qin, 2021
'''

'''
############### General Settings ################
'''
IMG_SIZE = (640, 640)       # (1792, 1792)
DATA_ROOT = '/mnt/qinlikun/dataset/DAVIS/DAVIS-semisupervised/DAVIS-trainval'

# corrupted area has size of (MASK_RATIO * height, MASK_RATIO * width)
MASK_RATIO = 0.25

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

'''
############### Training Setting ################
'''

SAVE_DIR = '/mnt/qinlikun/inpainting'

# maximum steps
MAX_ITER = 50000
DECAY_STEPS = [30000, 40000]

EPOCH = 100

BATCH_SIZE = 8

# number of workers for dataloader
N_THREADS = 8

# learning rate
LR = 1e-4

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

