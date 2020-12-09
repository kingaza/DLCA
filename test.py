# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:51:03 2020

@author: hejiew
"""

import numpy as np
from utils.data_utils import load_image, load_label
from utils.data_utils import crop_patch, oversample
from utils.data_utils import augment, map_label
from model import network


data_dir = './minidata/'
train_name = ['10072B']
filenames = [data_dir + "{}.nii.gz".format(idx) for idx in train_name]
config = network.config

patient_labels = load_label(data_dir, train_name)
aneurysm_labels = oversample(config, patient_labels)
        
idx = 0
if idx >= len(aneurysm_labels):
    neg_sample_flag = True
    idx = np.random.randint(len(aneurysm_labels))
else:
    neg_sample_flag = False
    
aneurysm_label = aneurysm_labels[idx]
patient_idx = int(aneurysm_label[0])
size = aneurysm_label[4]

image_path = filenames[patient_idx]
mean = 0.0
std = 100.0
image = load_image(image_path, mean, std)
patient_label = patient_labels[patient_idx]    

print('image', image.shape)


crop_dict = crop_patch(image, aneurysm_label[1:], patient_label, neg_sample_flag, config)
# label mapping
sample = crop_dict["image_patch"]
print('sample', sample.shape)

coord = crop_dict["coord"]
aneurysm_label = crop_dict["aneurysm_label"]
patient_label = crop_dict["patient_label"]

sample, aneurysm_label, patient_label, coord = augment(sample, aneurysm_label, patient_label, coord)

label = map_label(config, aneurysm_label, patient_label)
sample = sample.astype(np.float32)


