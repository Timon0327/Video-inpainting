import os
import sys
sys.path.append('..')
import cfgs.config_ytb as config

for j in range(1, 9):

    test_dir = os.path.join('/mnt/qinlikun/dataset/test/ytb', str(j), 'frames')

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


    # test_dir = os.path.join('/home/cap/dataset/demo_ytb/', str(j), 'flow')
    #
    # f_list = os.listdir(test_dir)
    # for one in f_list:
    #     name = one.split('.')[0]
    #     os.rename(os.path.join(test_dir, one), os.path.join(test_dir, '0' + one))
    #
    # f_list = os.listdir(test_dir)
    # flos = []
    # rflows = []
    # for one in f_list:
    #     if 'rflo' in one:
    #         rflows.append(one)
    #     else:
    #         flos.append(one)
    # print(flos)
    # flos.sort()
    # rflows.sort()
    # for i, one in enumerate(flos):
    #     name = one.split('.')[0]
    #     affix = one.split('.')[1]
    #     os.rename(os.path.join(test_dir, one), os.path.join(test_dir, '%05d.' % (i) + affix))
    #
    # for i, one in enumerate(rflows):
    #     name = one.split('.')[0]
    #     affix = one.split('.')[1]
    #     os.rename(os.path.join(test_dir, one), os.path.join(test_dir, '%05d.' % (i+1) + affix))