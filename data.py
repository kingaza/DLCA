
import os
import time
import collections
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.data_utils import load_image, load_label, map_label
from utils.data_utils import oversample, augment_image_label, crop_patch, flip


class TrainDataset(Dataset):
    def __init__(self, data_dir, data_names, config, aug_factor):

        self.config = config
        self.aug_factor = aug_factor
        # 0.3
        self.neg_ratio = config['neg_ratio']
        self.pad_value = config["pad_value"]
        self.data_names = data_names

        self.patient_labels = load_label(data_dir, self.data_names)
        self.aneurysm_labels = oversample(config, self.patient_labels, pos_aug=False)
        self.filenames = [os.path.join(data_dir, "{}.nii.gz".format(idx)) for idx in self.data_names]
    
    
    def __getitem__(self, idx):
        np.random.seed(int(str(time.time() % 1)[2:7]))

        if idx >= len(self.aneurysm_labels):
            neg_sample_flag = True
            idx = np.random.randint(len(self.aneurysm_labels))
        else:
            neg_sample_flag = False

        aneurysm = self.aneurysm_labels[idx]
        patient_idx = int(aneurysm[0])
        patient = self.patient_labels[patient_idx]
        aneurysm = aneurysm[1:]

        image_path = self.filenames[patient_idx]
        offset = 0.0
        scale = 1.0
        image = load_image(image_path, offset, scale)
        
        # rotate image and labels as well
        image_aug, aneurysm_aug, patient_aug = augment_image_label(image, aneurysm, patient, self.aug_factor)   
        # image_aug, aneurysm_aug, patient_aug = image, aneurysm, patient
        
        # crop sample
        crop_dict = crop_patch(image_aug, aneurysm_aug, patient_aug, neg_sample_flag, self.config)
        sample = crop_dict["image_patch"]
        coord  = crop_dict["coord"]
        aneurysm_label = crop_dict["aneurysm_label"]
        patient_label  = crop_dict["patient_label"]
        # print('  label ->', aneurysm_label[1:4], ', size =', aneurysm_label[4])
        
        # further flip augmentation
        sample, aneurysm_label, patient_label, coord = flip(sample, aneurysm_label, patient_label, coord)
    
        label = map_label(self.config, aneurysm_label, patient_label)
        sample = sample.astype(np.float32)

        return torch.from_numpy(sample), torch.from_numpy(label), coord, image_path

    def __len__(self):
        return int(len(self.aneurysm_labels) / (1 - self.neg_ratio))



class TestDataset(Dataset):
    def __init__(self, image_dir, test_name, config, split_comber=None):

        self.max_stride = config['max_stride']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        self.data_names = [test_name]
        self.filenames = [os.path.join(image_dir, '{}.nii.gz'.format(idx)) for idx in self.data_names]

    def __getitem__(self, idx, split=None):
        # t = time.time()
        # np.random.seed(int(str(t % 1)[2:7]))
        np.random.seed(3)  

        offset = 0. #-535.85
        scale = 1. #846.87
        imgs = load_image(self.filenames[idx], offset, scale)
        
        nz, nh, nw = imgs.shape[1:]
        pz = int(np.ceil(float(nz) / self.stride)) * self.stride
        ph = int(np.ceil(float(nh) / self.stride)) * self.stride
        pw = int(np.ceil(float(nw) / self.stride)) * self.stride
        imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                        constant_values=self.pad_value)

        xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, int(imgs.shape[1] / self.stride)),
                                    np.linspace(-0.5, 0.5, int(imgs.shape[2] / self.stride)),
                                    np.linspace(-0.5, 0.5, int(imgs.shape[3] / self.stride)), indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
        imgs2, nzhw = self.split_comber.split(imgs)
        coord2, nzhw2 = self.split_comber.split(coord,
                                                side_len=int(self.split_comber.side_len / self.stride),
                                                max_stride=int(self.split_comber.max_stride / self.stride),
                                                margin=int(self.split_comber.margin / self.stride))
        assert np.all(nzhw == nzhw2)

        print('coord', coord.shape, '-->', coord2.shape)
        print('images', imgs.shape, '-->', imgs2.shape)

        imgs2 = imgs2.astype(np.float32)
        return torch.from_numpy(imgs2), torch.from_numpy(coord2), np.array(nzhw)

    def __len__(self):
        
        return len(self.data_names)


def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]


