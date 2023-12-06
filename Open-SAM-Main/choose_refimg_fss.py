import os
import cv2
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

top_list = [0,1,2,3]
root = '/data/tanglv/data/fss-te/'
datasets = ['fold0','fold1','fold2','fold3','fss','perseg']

for dataset in datasets:
    gts_dir = os.path.join(root,dataset,'gts')
    for group in os.listdir(gts_dir):        
        for top in top_list:
            if os.path.exists(os.path.join(root,dataset,'ref'+str(top)+'.txt')):
                os.remove(os.path.join(root,dataset,'ref'+str(top)+'.txt'))


for dataset in tqdm(datasets):
    gts_dir = os.path.join(root,dataset,'gts')
    a2s_dir = os.path.join(root,dataset,'a2s')
    
    for group in tqdm(os.listdir(gts_dir)):
        if group.startswith('.'): continue
        gts_group_dir = os.path.join(root,dataset,'gts', group)
        a2s_group_dir = os.path.join(root,dataset,'a2s', group)
        
        pairs = []
        for file in os.listdir(gts_group_dir):
            gt_path = os.path.join(root,dataset,'gts', group,file)
            a2s_path = os.path.join(root,dataset,'a2s', group,file)

            gt = cv2.imread(gt_path,0)
            a2s = cv2.imread(a2s_path,0)                
            mae = np.abs(gt-a2s).mean()      
            #print(file)          
            pairs.append((mae.item(),file[:-4]))
            
        pairs = sorted(pairs, key=lambda x:x[0], reverse=True)
        
        # print(pairs)
        # raise NameError
        for top in top_list:
            with open(os.path.join(root,dataset,'ref'+str(top)+'.txt'),'a') as f:
                f.write(group+' '+ pairs[top][1]+'\n')









