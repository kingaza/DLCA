# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:51:03 2020

@author: hejiew
"""

import numpy as np
from scipy.ndimage import rotate, zoom

from utils.data_utils import load_image, load_label
from utils.data_utils import crop_patch, oversample
from utils.data_utils import augment, map_label
from model import network


data_dir = './train_data/ADAM2020/'
train_name = ['10072B']
filenames = [data_dir + "{}.nii.gz".format(idx) for idx in train_name]
config = network.config

patient_labels = load_label(data_dir, train_name)
aneurysm_labels = oversample(config, patient_labels)
        
idx = np.random.choice(len(aneurysm_labels))
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

print('-------- Select Image --------')
print('idx#{} in image'.format(idx), image.shape[1:])
print('aneurysm label', aneurysm_label[1:])
print('patient label', patient_label)


# Do Rotate or Zoom here
angle1 = np.random.uniform(-1,1) * 45
angle1 = 90
rotmat = np.array([[np.cos(angle1/180*np.pi), -np.sin(angle1/180*np.pi)],
                   [np.sin(angle1/180*np.pi), np.cos(angle1/180*np.pi)]])

assert len(image.shape)==4
image = rotate(image,angle1,axes=(2,3),reshape=False)
size = np.array(image.shape[2:4]).astype('float')
for i, box in enumerate(patient_label):
    assert box.size==4
    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
    # patient_label[i] = box
assert len(aneurysm_label)==5
aneurysm_label[2:4] = np.dot(rotmat,aneurysm_label[2:4]-size/2)+size/2    
print('-------- After Rotate --------')
print('rotate {:.3f} degree in image'.format(angle1), image.shape[1:])
print('aneurysm label', aneurysm_label[1:])
print('patient label', patient_label)

crop_dict = crop_patch(image, aneurysm_label[1:], patient_label, neg_sample_flag, config)
# label mapping
sample = crop_dict["image_patch"]

coord = crop_dict["coord"]
aneurysm_label = crop_dict["aneurysm_label"]
patient_label = crop_dict["patient_label"]
print('-------- After Crop --------')
print('sample', sample.shape[1:])
print('aneurysm label', aneurysm_label)
print('patient label', patient_label)

sample, aneurysm_label, patient_label, coord = augment(sample, aneurysm_label, patient_label, coord)
print('-------- After Augment --------')
print('sample', sample.shape[1:])
print('aneurysm label', aneurysm_label)
print('patient label', patient_label)

label = map_label(config, aneurysm_label, patient_label)
sample = sample.astype(np.float32)


