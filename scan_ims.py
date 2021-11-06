import cv2
import os
import numpy as np
import magic
import re

res_set = {}

train_path = "/mnt/a/data/256_object/256_ObjectCategories/"
train_dirs = os.listdir(train_path)
im_paths = []

for d in train_dirs:
    this_dir = os.listdir(train_path+d)
    for im in this_dir:
        im_paths.append(train_path + d + '/' + im)
    

res_counts = {}
for im in im_paths:
    image = magic.from_file(im)
    im_shape = re.search('(\d+)x(\d+)', image)
    if im_shape == None:
        continue
    im_shape = im_shape.groups()
    if im_shape not in res_counts.keys():
        res_counts[im_shape] = 0

    res_counts[im_shape] += 1


maxcount = 0
maxres   = ()
for key in res_counts.keys():
    print(str(key) + " : " + str(res_counts[key]))
    if res_counts[key] > maxcount:
        maxcount  = res_counts[key]
        maxres    = key

print(str(maxres) + " : " + str(res_counts[maxres]))