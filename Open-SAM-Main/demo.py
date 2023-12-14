names = ['2007_005469', '2007_009320', '2008_000919' ,'2009_000039']
root = '/data/tanglv/data/openvoc-te/VOC2012/gts'

import cv2
import numpy as np
import os

for name in names:
    path = os.path.join(root,name+'.png')
    print(path)
    gt = cv2.imread(path, 0)
    print(np.unique(gt))

