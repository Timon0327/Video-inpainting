'''
#  filename: fix_model.py
#  fix keys in checkpoint
#  By Likun Qin, 2021
'''
import torch
import copy
import os

model_dir = '/home/cap/project/inpainting'


def fix(model):
    '''
    @param model: model path
    '''
    ckpt = torch.load(os.path.join(model_dir, model))
    state_dict = ckpt['flownetcg']
    result = copy.deepcopy(state_dict)
    for key in list(state_dict):
        values = state_dict[key]
        new_key = key[7:]
        del result[key]
        result[new_key] = values

    print(model, ' done!')
    ckpt['flownetcg'] = result
    filename = model.split('.')[0] + 'new.pt'
    torch.save(ckpt, os.path.join(model_dir,filename))

if __name__ == '__main__':

    fix('flownetcg_11000.pt')