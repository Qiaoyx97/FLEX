import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os
from glob import glob
import openslide

#Generate numpy files corresponding to the original pathology images (.svs).
image_np_dir = 'np/'
#img_list.txt is a file that stores patient IDs and original image paths by row.
#e.g. s1+'\t'+raw_image/s1.svs
img_list = {}
with open('img_list.txt', 'r') as f:
    for l in f.readlines():
        img_list[l.split('\t')[0]] = l.split('\t')[1][:-1]
for key,value in tqdm(img_list.items()):   
    image_name = key
    image = openslide.open_slide(value)
    [w,h] = image.level_dimensions[0]
    image_np = np.array(image.read_region((0,0),0,image.dimensions))
    np.save(image_np_dir+image_name, image_np)



#Segmentation of tissue areas.
save_path = 'tissue/'
nlist = glob(image_np_dir)
for i in tqdm(range(len(nlist))):
    img = np.load(nlist[i])
    name = nlist[i].split('/')[-1].split('.')[0]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    raw = cv2.resize(hsv, (1024,1024))
    raw = cv2.GaussianBlur(raw, (21,21), 0)
    _, raw_M = cv2.threshold(raw[:,:,1], 20, 255, cv2.THRESH_BINARY)
    retval, label, stats, centroids  = cv2.connectedComponentsWithStats(raw_M.astype(np.int8), connectivity=8)
    select = []
    select_label = []
    for i in range(1,len(stats)):
        if stats[i,2] * stats[i,3] > 1024*1024*0.005:
            select_label.append(i)
            select.append(stats[i,:].tolist())
    for j in range(1,len(stats)):
        if j not in select_label:
            label[label == j] = 0
        else:
            label[label == j] = 1
    s = name
    for se in select:
        s = s + '\t' + str(se)
    np.save(save_path+name, label)



#Generate .npy files for image input.
tissue_path = 'tissue/*.npy'
save_path = 'img/'

plist = glob(tissue_path)
for p in tqdm(range(len(plist))):
    tissue = np.load(plist[p])
    name = plist[p].split('/')[-1].split('.npy')[0]
    raw = np.load(glob(rpath + name + '*')[0])
    raw = cv2.resize(raw, (1024,1024))
    tissue = np.stack([tissue, tissue, tissue, tissue], axis=-1)
    tr = tissue * raw
    np.save(save_path + name, tr)
