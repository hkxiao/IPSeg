import os
import cv2
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import torch
from kmeans_pytorch import kmeans
from torch.nn import functional as F
import argparse

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_arguments():
    #CUDA_VISIBLE_DEVICES=0 python persam_dino_kmeans.py
    parser = argparse.ArgumentParser()

    # prompt setting
    parser.add_argument('--ptopk', type=int, default=32)
    parser.add_argument('--pt', type=int, default=4)
    parser.add_argument('--ntopk', type=int, default=32)
    parser.add_argument('--nt', type=int, default=4)

    #trick setting
    parser.add_argument('--erosion', action="store_true")
    parser.add_argument('--oneshot', action='store_true')
    
    #sd setting
    parser.add_argument('--sd_weight', type=float, default=0.)
    parser.add_argument('--sd_layer_weight', type=str, default="1,1,1")
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--copca', action='store_true')
    
    parser.add_argument('--datasets', nargs='+', type=str, help='Input a list of integers')
    args = parser.parse_args()
    args.sd_layer_weight = args.sd_layer_weight.split(',')
    args.sd_layer_weight = [float(x) for x in args.sd_layer_weight]
    
    return args


def get_point(correlation_map, topk, t): # N k H W
    fix_randseed(0)        
    correlation_max = torch.max(correlation_map, dim=1)[0] # N H W
    ranged_index = torch.argsort(torch.flatten(correlation_max, -2), 1, descending=True) #N HW
    coords = torch.stack([ranged_index[:,:topk]%60,ranged_index[:,:topk]/60],-1) #N topk 2
    centers = []
    for i in range(coords.shape[0]):
        center = kmeans(coords[i],K=t, max_iters=20) #t 2
        centers.append(center)
    max_centers = torch.stack(centers,dim=0) #N t 2
    
    return max_centers 
args = get_arguments()
top_list = [0]
root = '/data/tanglv/data/fss-te/'
datasets = args.datasets

global unseen_ids
    if 'coco-stuff' in args.data:
        unseen_ids = [34, 41, 100, 57, 87, 33, 25, 21, 149, 172, 169, 124, 148, 106, 145]
        #frisbee, skateboard, cardboard, carrot, scissors, suitcase, giraffe, cow, road, wall concrete, tree, grass, river, clouds, playingfield
    if 'VOC' in args.data:
        unseen_ids = [37,52,94,112,150]
        #potted plant, Tv-monitor, cow, sofa, train
    if 'pcontext' in args.data:
        unseen_ids = [14,19,33,48]
        #cat cow moterbikt sofa


for dataset in datasets:
    gts_dir = os.path.join(root,dataset,'gts')
    for group in os.listdir(gts_dir):        
        for top in top_list:
            if os.path.exists(os.path.join(root,dataset,'match_point_ref'+str(top)+'.txt')):
                os.remove(os.path.join(root,dataset,'match_point_ref'+str(top)+'.txt'))


for dataset in tqdm(datasets):
    gts_dir = os.path.join(root,dataset,'gts')
    a2s_dir = os.path.join(root,dataset,'a2s')
    feat_dir = os.path.join(root,dataset,'sd_raw+dino_feat')
    
    for group in tqdm(os.listdir(gts_dir)):
        if group.startswith('.'): continue
        gts_group_dir = os.path.join(root,dataset,'gts', group)
        a2s_group_dir = os.path.join(root,dataset,'a2s', group)
        feat_group_dir = os.path.join(root,dataset,'sd_raw+dino_feat', group)
        
        all_value_list = []
        for file in tqdm(os.listdir(gts_group_dir)):
            ref_gt_path = os.path.join(root,dataset,'gts', group,file)
            ref_a2s_path = os.path.join(root,dataset,'a2s', group,file)
            ref_feat_path = os.path.join(root,dataset,'sd_raw+dino_feat', group, file.replace('png','pth'))
                    
            # load ref_mask
            ref_mask = cv2.imread(ref_a2s_path)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)
            ref_mask = torch.tensor(ref_mask).cuda().unsqueeze(0).unsqueeze(0).to(torch.float32) # 1 1 H W
            ref_mask = F.interpolate(ref_mask, size=(60,60), mode="nearest") #1 1 h w
            if args.oneshot: ref_mask[ref_mask!=0] = 1
            else: ref_mask = ref_mask / 255.0
            if args.erosion:
                max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  
                ref_mask = -max_pool(-ref_mask)
            ref_mask = ref_mask.squeeze() # h w
            
            # load ref_feat
            ref_all_feat = torch.load(ref_feat_path, map_location='cuda')
            sd_feat, dino_feat = ref_all_feat['sd_feat'], ref_all_feat['dino_feat'] #  [1 1 3600 768]  
            ref_feat1 = dino_feat.reshape(60,60,768)  
            ref_feat2 = {}
            for k,v in sd_feat.items():
                #[1 1280 15 15] [1 1280 30 30] [1 640 60 60]-> [60 60 1280] [60 60 1280] [60 60 640] 
                if k!= 's2': ref_feat2[k]= F.interpolate(v,size=(60,60),mode='nearest').squeeze().permute(1,2,0)
                if args.pca: ref_feat2[k] = pca(ref_feat2[k].view(3600,-1)).view(60,60,-1)
            
            #get target feat1
            target_feat1 = ref_feat1[ref_mask>0.5]  # N C   
            target_feat1 = target_feat1.mean(0).unsqueeze(0) # N C -> 1 C
            target_feat1 = target_feat1 / target_feat1.norm(dim=-1, keepdim=True) # 1 C
            
            #get target feat2  
            target_feat2 = {}
            for k,v in ref_feat2.items():
                target_feat2[k] = v[ref_mask>0.5] # N C
                target_feat2[k] = target_feat2[k].mean(0).unsqueeze(0) # N C -> 1 C
                target_feat2[k] = target_feat2[k] / target_feat2[k].norm(dim=-1, keepdim=True) # 1 C
            
            # start testing
            print('======> Enumerate test imgs')
            all_value = 0
            for _file in tqdm(sorted(os.listdir(gts_group_dir))):            
                # Load test path
                test_gt_path = os.path.join(root,dataset,'gts', group,_file)
                test_a2s_path = os.path.join(root,dataset,'a2s', group,_file)
                test_feat_path = os.path.join(root,dataset,'sd_raw+dino_feat', group, _file.replace('png','pth'))
                
                # Load test mask
                test_mask = cv2.imread(test_gt_path)
                test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
                test_mask = cv2.resize(test_mask,(1024,1024))
                test_mask = torch.tensor(test_mask).cuda().unsqueeze(0).unsqueeze(0) # [1 1 H W]
                test_mask[test_mask>0] =1
                test_mask = torch.squeeze(test_mask)
                
                # Load test feat
                test_all_feat = torch.load(test_feat_path, map_location='cuda')
                sd_feat, dino_feat = test_all_feat['sd_feat'], test_all_feat['dino_feat']
                
                test_feat1 = dino_feat.reshape(60,60,768)  
                test_feat2 = {}
                for k,v in sd_feat.items():
                    #[1 1280 15 15] [1 1280 30 30] [1 640 60 60]-> [60 60 1280] [60 60 1280] [60 60 640] 
                    if k!= 's2': test_feat2[k]= F.interpolate(v,size=(60,60),mode='nearest').squeeze().permute(1,2,0)
                    if args.pca: test_feat2[k] = pca(test_feat2[k].view(3600,-1)).view(60,60,-1)

                # Cosine similarity 1
                test_feat1  = test_feat1.permute(2,0,1)
                C, h, w = test_feat1.shape
                test_feat1 = test_feat1 / test_feat1.norm(dim=0, keepdim=True) # C h w
                test_feat1 = test_feat1.reshape(C, h * w) # C hw
                sim1 = target_feat1 @ test_feat1 # [1 C] @ [C hw]
                sim1 = sim1.reshape(1, 1, h, w)
                
                # Cosine similarity 2
                sim2 = 0
                for i,k in enumerate(test_feat2.keys()):
                    # s5 s4 s3
                    test_feat2[k]  = test_feat2[k].permute(2,0,1)
                    C, h, w = test_feat2[k].shape
                    test_feat2[k] = test_feat2[k] / test_feat2[k].norm(dim=0, keepdim=True) # C h w
                    test_feat2[k] = test_feat2[k].reshape(C, h * w) # C hw
                    sim2_tmp = target_feat2[k] @ test_feat2[k] # [1 C] @ [C hw]
                    sim2_tmp = sim2_tmp.reshape(1, 1, h, w)            
                    sim2 += sim2_tmp.to(torch.float32) * args.sd_layer_weight[i]

                #get composed sim
                sim = sim1 + sim2 * args.sd_weight
            
                #get point prompt     
                p_coords, n_coords = get_point(sim,args.ptopk,args.pt), get_point(1-sim,args.ntopk,args.nt) # [1 t 2]
                p_coords, n_coords = p_coords.view(args.pt,2),  n_coords.view(args.nt,2) # [t 2]
                
                #clear prompt
                tmp_p_coords, tmp_n_coords = torch.empty(0,2).cuda(), torch.empty(0,2).cuda()
                for p_coord in p_coords:            
                    if not torch.isnan(p_coord).any().item(): tmp_p_coords = torch.cat([tmp_p_coords, p_coord.unsqueeze(0)])
                for n_coord in n_coords:            
                    if not torch.isnan(n_coord).any().item(): tmp_n_coords = torch.cat([tmp_n_coords, n_coord.unsqueeze(0)])
                
                p_coords=tmp_p_coords 
                n_coords=tmp_n_coords 
                
                #input prompt
                p_coords, n_coords = p_coords*1024/60+1024/120, n_coords*1024/60+1024/120
                p_coords, n_coords = p_coords.to(torch.long), n_coords.to(torch.long)

                p_coords = torch.clip(p_coords,0,1023)
                n_coords = torch.clip(n_coords,0,1023)
                
                p_value = torch.sum(test_mask[p_coords[:,0],p_coords[:,1]])
                n_value = test_mask[n_coords[:,0],n_coords[:,1]]
                n_value = n_value.shape[0] - torch.sum(n_value)
                
                value = p_value + n_value
                all_value += value.item()
            all_value_list.append((all_value,file))

        all_value_list = sorted(all_value_list, key=lambda x:x[0], reverse=True)
        # print(all_value_list)
        # print("/n/n")
        # raise NameError
        
        for top in top_list:
            with open(os.path.join(root,dataset,'match_point_ref'+str(top)+'.txt'),'a') as f:
                f.write(group+' '+ all_value_list[top][1][:-4]+'\n')