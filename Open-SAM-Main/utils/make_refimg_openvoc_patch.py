import os
import cv2
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

seed=7
random.seed(seed)

root = '/data/tanglv/data/fss-te/'
#datasets = ['coco-stuff','pcontext','VOC2012']
datasets = ['coco-stuff']

for dataset in datasets:
    img_root = os.path.join(root,dataset,'imgs')
    gt_root = os.path.join(root,dataset,'gts')
    ref_root = os.path.join(root,dataset,'refimg'+str(seed))
    
    Path(ref_root).mkdir(parents=True, exist_ok=True)

    unique_list = []
    for file in os.listdir(gt_root):
        gt = cv2.imread(os.path.join(gt_root,file),0)
        unique_list.extend(np.unique(gt))
    unique_list = np.unique(unique_list)

    if dataset=='coco-stuff': unique_list=unique_list[:-1]
    if dataset=='pcontext': unique_list=unique_list[:-1]
    if dataset=='VOC2012': unique_list=unique_list[1:-1]
    print(dataset,len(unique_list))
    
    file_list = sorted(os.listdir(img_root))
    for i in tqdm(unique_list):    
        random.shuffle(file_list)
        for file in tqdm(file_list):
            img = cv2.imread(os.path.join(img_root,file))
            img2 = img.copy()
            img3 = img.copy()
            gt = cv2.imread(os.path.join(gt_root,file.replace('jpg','png')),0)
            
            if np.any(gt==i):
                img[gt!=i]=0
                img = np.max(img,axis=-1)
                x, y = np.nonzero(img)
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                
                img2[gt!=i]=0
                #cv2.imwrite(ref_root+'ref_'+str(i)+'_matting.jpg', img2[x_min:x_max+1,y_min:y_max+1])
                cv2.imwrite(ref_root+'/ref_'+str(i)+'.jpg', img3[x_min:x_max+1,y_min:y_max+1])
                break
            







