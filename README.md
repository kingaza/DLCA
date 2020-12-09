# DLCA
![detection_visualization](vis_cta.gif)
#### left: input (3D volume)
#### right: output (3D volume. Blue boxes：Predictions; Red box：GroundTrue label）

### Table of contents

  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)


## Introduction

This repository contains the source code and the trained model for Deep Learning for Detecting Cerebral Aneurysms on CT or MR Angiography.

## Prerequisites
- Ubuntu
- Python 3.x
- Pytorch 1.6
- NVIDIA GPU + CUDA CuDNN

This repository has been tested on NVIDIA TITAN Xp. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

## Installation
* Clone this repo:
```bash
git clone https://github.com/CTA-detection/DLCA.git
cd DLCA
```
* Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Preprocess
* Run command as below.
```bash
python ./utils/pre_process.py --input="./raw_data/" --output="./train_data/"
```

### 2. Train 
* Run command as below.
```bash
python train.py -j=16 -b=12 --input="train_data/" --output="./checkpoint/"
```

### 3. Inference 
* Click the [checkpoint link(GoogleDrive)](https://drive.google.com/drive/folders/138_EpuZaMB0sS_dVmO0ux6_07sFfwRKZ?usp=sharing) to download trained model into "./checkpoint".
* Click the [data link(GoogleDrive)](https://drive.google.com/file/d/1M76tVZp-dqW9COlESnh0n8iuii5PKiS8/view?usp=sharing) to download test image "brain_CTA.nii.gz" 
* Run command as below.
```bash
# an example with the image named "brain_CTA.nii.gz"
python inference.py -j=1 -b=1 --checkpoint="./checkpoint/trained_model.ckpt" --input="./test_image/brain_CTA" --output="./prediction/brain_CTA"
```

### 4. Fine-tune from CTA to MRA 
* Download data of [ADAM2020 challenge](http://adam.isi.uu.nl/)
* Pre-process data
```bash
python ./utils/preprocess_adam2020.py --input="./adam2020/" --output="./train_data/"
```
* fine-tune with pretrained model
```bash
python train.py -j=16 -b=12 --resume="./checkpoint/trained_model.ckpt" --input="train_data/" --output="./checkpoint/" 
```
Note: rescale the input data, and make the value of aneurysm region to be normalized in [0,1]
## Results

| Sensitivity | False Positive per case |
|:-------------:|:-------------:|
| 97.5% | 13.8| 

