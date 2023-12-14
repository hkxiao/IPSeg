import os
import cv2
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

seed=12
random.seed(seed)

root = '/data/tanglv/data/openvoc-te/'
#datasets = ['coco-stuff','pcontext','VOC2012']
datasets = ['coco-stuff']




for dataset in datasets:
    
    if 'coco-stuff' in dataset:
        unseen_ids = [33, 40, 99, 56, 86, 32, 24, 20, 148, 171, 168, 123, 147, 105, 144]
        #frisbee, skateboard, cardboard, carrot, scissors, suitcase, giraffe, cow, road, wall concrete, tree, grass, river, clouds, playingfield
    if 'VOC2012' in dataset:
        unseen_ids = [16384, 16512, 8404992, 49152, 8437760]
        #potted plant, Tv-monitor, sheep, sofa, train
    if 'pcontext' in dataset:
        unseen_ids = [14, 19, 33, 48]
        #cat cow moterbikt sofa    
    

    img_root = os.path.join(root,dataset,'imgs')
    gt_root = os.path.join(root,dataset,'gts')
    ref_root = os.path.join(root,dataset,'refimg'+str(seed))
    
    Path(ref_root).mkdir(parents=True, exist_ok=True)

    # if dataset=='coco-stuff': unique_list=unique_list[:-1]
    # if dataset=='pcontext': unique_list=unique_list[:-1]
    # if dataset=='VOC2012': unique_list=unique_list[1:-1]
    # print(dataset,len(unique_list),unique_list)
    

    file_list = sorted(os.listdir(img_root))
    for i in tqdm(unseen_ids):    
        random.shuffle(file_list)
        for file in tqdm(file_list):
            img = cv2.imread(os.path.join(img_root,file))
            img2 = img.copy()
            img3 = img.copy()
            
            if dataset == 'VOC2012': 
                gt = cv2.imread(os.path.join(gt_root,file.replace('jpg','png')),1).astype(np.uint32)
                gt = gt[:,:,0] + gt[:,:,1]*256 + gt[:,:,2]*256*256
            else: gt = cv2.imread(os.path.join(gt_root,file.replace('jpg','png')),0)
            
            if np.any(gt==i):
                img[gt!=i]=0
                img = np.max(img,axis=-1)
                x, y = np.nonzero(img)
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                
                img2[gt!=i]=0
                #cv2.imwrite(ref_root+'/ref_'+str(i)+'_matting.jpg', img2[x_min:x_max+1,y_min:y_max+1])
                cv2.imwrite(ref_root+'/ref_'+str(i)+'.jpg', img3[x_min:x_max+1,y_min:y_max+1])
                break
            







