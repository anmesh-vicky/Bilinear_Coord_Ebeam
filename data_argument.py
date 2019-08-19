#!/usr/bin/env python
# coding: utf-8

### import libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import hamming

##### data split into train, test and validation folder

import split_folders
split_folders.ratio('./data_str', output="./data", seed=1337, ratio=(.70, .15, .15)) # default values


#### Data Argumentor

import Augmentor
import shutil
##No of Augmented Images  per class
import os
n=50
No_train = 6000*.7  
No_test = 6000*.15
No_val = 6000*.15
directory = ["train/","test/","val/"]
base = "./data/"
for loc in directory:
    if loc =="train/":
        n = No_train
    elif loc =="test/":
        n = No_test
    else:
        n = No_val
    for dir in os.listdir(base+loc):
        p= Augmentor.Pipeline(base+loc+dir)
        p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
        p.rotate_random_90(probability=0.7)
        num_of_samples = int(n)
        p.sample(num_of_samples)
        
        src_files = os.listdir(base+loc+dir+"/output")
        src = base+loc+dir+"/output"
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            shutil.copy(full_file_name, base+loc+dir)

        shutil.rmtree(base+loc+dir+"/output")      
