import os
import cv2
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import shutil

dataroot = '/data/tanglv/data/openvoc-te'
outroot = 'outputs/openvoc/'
datasets = ['coco-stuff']
outlist = ['x_refimg0_x_x_erosion_32_4_32_4_SD_0.1_[0.3, 0.2, 0.1]_0_100',
           'x_refimg1_x_x_erosion_32_4_32_4_SD_0.1_[0.3, 0.2, 0.1]_0_100',
           'x_refimg2_x_x_erosion_32_4_32_4_SD_0.1_[0.3, 0.2, 0.1]_0_100',          
           'x_refimg3_x_x_erosion_32_4_32_4_SD_0.1_[0.3, 0.2, 0.1]_0_100',
           ]

for dataset in datasets:
    dataset_dir = os.path.join(dataroot,dataset)
    out_dir = os.path.join(outroot,dataset)

    info_dict = {}
            
    for out in outlist:
        txt = open(os.path.join(out_dir,out,'log.txt'),'r')
        for line in txt.readlines():
            if line.startswith('All'): continue
            context = line.split(' ')
            if len(context) != 3: continue
            group, ref_name, miou = context[0], context[1], float(context[2][:-1])
            if (not group in info_dict.keys()) or miou > info_dict[group][1]:
                info_dict[group] = (ref_name, miou)
     
    composd_dir = os.path.join(dataset_dir,'refimg_composed')
    if os.path.exists(composd_dir): shutil.rmtree(composd_dir)
    # Path(composd_dir).mkdir(parents=True,exist_ok=True)
    shutil.copytree(os.path.join(dataset_dir+'/refimg0'),composd_dir)
    
    miou = 0
    for k, v in info_dict.items():
        miou += v[1]
        # print(composd_dir+'ref_'+k+'.jpg')
        shutil.copyfile(os.path.join(dataset_dir,v[0],'ref_'+k+'.jpg'), composd_dir+'/ref_'+k+'.jpg')
        shutil.copyfile(os.path.join(dataset_dir,v[0],'ref_'+k+'.png'), composd_dir+'/ref_'+k+'.png')
        shutil.copyfile(os.path.join(dataset_dir,v[0],'ref_'+k+'.pth'), composd_dir+'/ref_'+k+'.pth')
        
    print(dataset,'miou',miou/len(info_dict.keys()))
    
    

        
            







