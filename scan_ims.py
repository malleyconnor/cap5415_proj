import cv2
import os
import numpy as np
import magic
import re
import shutil
from time import sleep
from tqdm import tqdm

res_set = {}

res_to_get = ('640', '480')
train_path = "/mnt/a/data/256_object/256_ObjectCategories/"
train_dirs = os.listdir(train_path)
im_paths = []

print("Scanning image directory contents...")
for i in tqdm(range(len(train_dirs))):
    this_dir = os.listdir(train_path+train_dirs[i])
    for im in this_dir:
        im_paths.append(train_path + train_dirs[i] + '/' + im)
    

res_counts = {}
print("Relocating %s res images.."%(str(res_to_get)))
for i in tqdm(range(len(im_paths))):
    im = im_paths[i]
    image = magic.from_file(im)
    im_shape = re.findall('(\d+)x(\d+)', image)
    if len(im_shape) < 2:
        continue
    im_shape = re.findall('(\d+)x(\d+)', image)[1]
    if im_shape not in res_counts.keys():
        res_counts[im_shape] = 0

    res_counts[im_shape] += 1
#    if im_shape == res_to_get:
#        try:
#            os.mkdir('/mnt/a/data/256_object/640_480')
#        except (OSError):
#            ...
#
#        shutil.move(im, '/mnt/a/data/256_object/640_480/' + im.split('/')[-1])
    



maxcount = 0
maxres   = ()
for key in res_counts.keys():
    print(str(key) + " : " + str(res_counts[key]))
    if res_counts[key] > maxcount:
        maxcount  = res_counts[key]
        maxres    = key

print(str(maxres) + " : " + str(res_counts[maxres]))