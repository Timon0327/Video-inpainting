'''
#  filename: select_ytb.py
#  select 50% youtube dataset for training
'''
import os
import random
import shutil

percentage = 0.5
data_dir = '/home/cap/dataset/tiny_ytb'
backup_dir = '/home/cap/dataset/backup'

video_names = os.listdir(os.path.join(data_dir, 'JPEGImages'))
video_num = len(video_names)
random.shuffle(video_names)

if not os.path.exists(os.path.join(backup_dir, 'JPEGImages')):
    os.makedirs(os.path.join(backup_dir, 'JPEGImages'))
if not os.path.exists(os.path.join(backup_dir, 'feature')):
    os.makedirs(os.path.join(backup_dir, 'feature'))

choose_num = int(percentage * video_num)
for name in video_names[:choose_num]:
    img_dir = os.path.join(data_dir, 'JPEGImages', name)
    target_img = os.path.join(backup_dir, 'JPEGImages', name)
    shutil.move(img_dir, target_img)

    feature_dir = os.path.join(data_dir, 'feature', name)
    target_feature = os.path.join(backup_dir, 'feature', name)
    shutil.move(feature_dir, target_feature)

