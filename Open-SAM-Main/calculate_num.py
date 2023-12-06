import os
import cv2
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

dataroot = '/data/tanglv/data/fss-te'
outroot = 'outputs/fss-te/'
datasets = ['fold0','fold1','fold2','fold3','fss','P_fold0','P_fold1','P_fold2','P_fold3']

for dataset in datasets:
    dataset_dir = os.path.join(dataroot,dataset)
       
    f = open(os.path.join(dataset_dir,'num.txt'),'w')

    for group in os.listdir(dataset_dir+'/imgs'): 
        group_dir = os.path.join(dataset_dir,'imgs',group)
        imgs = os.listdir(group_dir)
        imgs = [img for img in imgs if img.endswith('jpg')]
        f.write(group + ' ' + str(len(imgs)) + '\n')
    
    f.close()
        
            







