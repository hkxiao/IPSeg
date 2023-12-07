import os
import cv2
from pathlib import Path
import random
import numpy as np

random_seed = 2
random.seed(random_seed)

data_path = '/data/tanglv/data/fss-te/perseg/'
img_root = data_path + 'imgs/'
gt_root = data_path + 'gts/'
ref_root = data_path + 'refimg'+str(random_seed)
Path(ref_root).mkdir(parents=True, exist_ok=True)

group_list = os.listdir(img_root)

for group in group_list:
    if group.startswith('.'): continue
    id = group
    group_path = os.path.join(img_root,group)
    files = os.listdir(group_path)
    file = files[random.randint(0,len(files)-1)]
    img = cv2.imread(os.path.join(img_root,group,file))
    img2 = img.copy()
    img3 = img.copy()
    gt = cv2.imread(os.path.join(gt_root,group,file.replace('jpg','png')))
    gt = cv2.resize(gt,img.shape[:-1][::-1])
    gt = np.max(gt,-1,keepdims=True)
    gt = np.concatenate([gt]*3,-1)
    
    a2s = cv2.imread(os.path.join(gt_root.replace('gts','a2s'),group,file.replace('jpg','png')))
    a2s = cv2.resize(a2s,img.shape[:-1][::-1])
    
    img[gt==0]=0
    img = np.max(img,axis=-1)
    x, y = np.nonzero(img)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    img2[gt==0]=0
    cv2.imwrite(ref_root+'/ref_'+id+'_matting.jpg', img2[x_min:x_max+1,y_min:y_max+1])
    cv2.imwrite(ref_root+'/ref_'+id+'.jpg', img3[x_min:x_max+1,y_min:y_max+1])
    cv2.imwrite(ref_root+'/ref_'+id+'.png', gt[x_min:x_max+1,y_min:y_max+1])
    cv2.imwrite(ref_root+'/ref_'+id+'_a2w.png', a2s[x_min:x_max+1,y_min:y_max+1])









