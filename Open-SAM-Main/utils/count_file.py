import os

root = '/data/tanglv/data/fss-te'
datasets = ['fss','P_fold0', 'P_fold1', 'P_fold2', 'P_fold3']

for dataset in datasets:
    img_dir = os.path.join(root,dataset,'imgs')
    feat_dir = os.path.join(root,dataset,'sd_raw+dino_feat')
    for group in os.listdir(img_dir):
        img_group_dir = os.path.join(img_dir,group)
        feat_group_dir = os.path.join(feat_dir,group)
        
        imgs = os.listdir(img_group_dir)
        feats = os.listdir(feat_group_dir)

        
        imgs = [x for x in imgs if '.jpg' in x]
        feats = [x for x in feats if '.pth' in x]
        

        if len(imgs)!=len(feats): 
            print(len(imgs), len(feats))
            print(dataset,group)
            print("False")
