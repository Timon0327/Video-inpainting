import cv2 as cv
import numpy as np
import cvbase as cvb
import os

flow_dir = '/home/cap/dataset/demo_ytb/flow'
save_dir = '/home/cap/dataset/demo_ytb/flow_visual'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

flows = os.listdir(flow_dir)
flows.sort()
for one in flows:
    name = one.split('.')[0]
    if 'rflo' in one:
        name = name + 'r'
    flow = cvb.read_flow(os.path.join(flow_dir, one))
    flow_img = cvb.flow2rgb(flow)
    cvb.show_flow(flow=flow, win_name=one, wait_time=600)
    # cv.imwrite(os.path.join(save_dir, name + '.jpg'), flow_img)