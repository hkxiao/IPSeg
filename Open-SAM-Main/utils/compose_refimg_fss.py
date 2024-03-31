import os
import cv2
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

dataroot = '/data/tanglv/data/fss-te'
outroot = 'outputs/fss-te/'
datasets = ['fold0','fold1','fold2','fold3']
outlist = [
           'match_point_ref0.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'ref_manual2.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'ref_manual3.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'ref0.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'ref1.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'ref2.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'ref3.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'x_refimg0_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'x_refimg1_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4',
           'x_refimg2_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4'
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
    
    f3 = open(os.path.join(dataset_dir,'num.txt'),'r')   
    num_dict = {}
    for line in f3.readlines():
        #print(line)
        group, num = tuple(line.split(' '))
        # print(group,num)
        num_dict[group] = int(num)
    
    f1 = open(os.path.join(dataset_dir,'ref_composed.txt'),'w')
    f2 = open(os.path.join(dataset_dir,'ref_composed_performance.txt'),'w')
    
    miou = 0
    nums = 0
    for k, v in info_dict.items():
        f1.write(k+' '+v[0]+'\n')
        f2.write(k+' '+v[0]+' '+str(v[1])+'\n')
        miou += v[1] * num_dict[k]
        nums += num_dict[k]
        
    print(dataset,'miou',miou/nums)
    f2.write('miou '+str(miou/nums)+'\n')
    f1.close()
    f2.close()
        
            







