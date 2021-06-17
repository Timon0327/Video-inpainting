import os
import sys
sys.path.append('..')
import cfgs.config_ytb as config

test_dir = os.path.join('/home/cap/dataset/demo_ytb/', 'frames')

f_list = os.listdir(test_dir)
for one in f_list:
    name = one.split('.')[0]
    os.rename(os.path.join(test_dir, one), os.path.join(test_dir, '0' + one))

f_list = os.listdir(test_dir)
f_list.sort()
print(f_list)
for i, one in enumerate(f_list):
    name = one.split('.')[0]
    os.rename(os.path.join(test_dir, one), os.path.join(test_dir, '%05d.jpg' % (i)))