'''
#  filename: config.py
#  hyper-parameters
#  By Likun Qin, 2021
'''

'''
############### General Settings ################
'''
IMG_SIZE = (800, 800)       # (1792, 1792)
DATA_ROOT = ''

# each image is divided by SLICE * SLICE patches
SLICE = 4

# how many batch are needed to process one frame (to save CUDA memory)
DIV = 1

# how many frames, before and after separately, are used to build temporal GCN
N = 4


'''
############### Training Setting ################
'''

# corrupted area has size of (MASK_RATIO * height, MASK_RATIO * width)
MASK_RATIO = 0.25


