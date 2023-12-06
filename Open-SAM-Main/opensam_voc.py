import numpy as np
import torch
from torch.nn import functional as F
import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from datetime import datetime
from show import *
from segment_anything import sam_model_registry, SamPredictor
from process_feat import process_feat
from kmeans_pytorch import kmeans
import random
from pathlib import Path

def co_pca(feat1, feat2, target_dim=256):
    size1  = feat1.shape[0]
    feat = torch.cat([feat1,feat2],0) # 2hw C
    
    #PCA
    mean = torch.mean(feat, dim=0, keepdim=True) #2hw C
    centered_feat = feat - mean #2hw C
    U, S, V = torch.pca_lowrank(centered_feat, q=target_dim)
    reduced_feat = torch.matmul(centered_feat, V[:, :target_dim]) # [2hw C] @ [C target_dim]
    
    #split
    return reduced_feat[:size1], reduced_feat[size1:]


def pca(feat, target_dim=256):
    mean = torch.mean(feat, dim=0, keepdim=True) #hw C
    centered_feat = feat - mean #hw C
    U, S, V = torch.pca_lowrank(centered_feat, q=target_dim)
    reduced_feat = torch.matmul(centered_feat, V[:, :target_dim]) # [hw C] @ [C target_dim]
    
    return reduced_feat

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def compute_iou(preds, target): #N 1 H W
    def mask_iou(pred_label,label):
        '''
        calculate mask iou for pred_label and gt_label
        '''

        pred_label = (pred_label>0.5)[0].int()
        label = (label>0.5)[0].int()

        intersection = ((label * pred_label) > 0).sum()
        union = ((label + pred_label) > 0).sum()
        return intersection / union
    
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + mask_iou(postprocess_preds[i],target[i])
    return iou.item() / len(preds)

def get_arguments():
    #CUDA_VISIBLE_DEVICES=0 python persam_dino_kmeans.py
    parser = argparse.ArgumentParser()
    
    #ref setting
    parser.add_argument('--ref_txt', default='x')
    parser.add_argument('--ref_img', default='x')
    parser.add_argument('--ref_sed', default='x')
    parser.add_argument('--ref_idx', default='x')
    
    #trick setting
    parser.add_argument('--oneshot', action='store_true')
    parser.add_argument('--matting', action='store_true')
    parser.add_argument('--erosion', action="store_true")
    parser.add_argument('--prompt_filter', action="store_true")
    parser.add_argument('--mask_filter', action="store_true")
    
    # prompt setting
    parser.add_argument('--ptopk', type=int, default=32)
    parser.add_argument('--pt', type=int, default=4)
    parser.add_argument('--ntopk', type=int, default=32)
    parser.add_argument('--nt', type=int, default=4)
    
    #sd setting
    parser.add_argument('--sd_weight', type=float, default=0.)
    parser.add_argument('--sd_layer_weight', type=str, default="1,1,1")
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--copca', action='store_true')
    
    #base setting
    parser.add_argument('--data', type=str, default='/data/tanglv/data/fss-te/fold0')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--outdir', type=str, default='fss-te')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--start', default=0,type=int)
    parser.add_argument('--end', default=-1,type=int)
    

    args = parser.parse_args()
    args.sd_layer_weight = args.sd_layer_weight.split(',')
    args.sd_layer_weight = [float(x) for x in args.sd_layer_weight]
    return args


def main():
    args = get_arguments()
    print("Args:", args)
    
    images_path = args.data + '/imgs/'
    
    #ref suffix
    suffix = args.ref_txt+'_'+args.ref_img + '_' + args.ref_sed + '_' + args.ref_idx
    
    #trick suffix
    if args.erosion: suffix += '_erosion'
    if args.oneshot: suffix += '_oneshot'
    if args.prompt_filter: suffix += '_prompt-filter'
    if args.mask_filter: suffix += '_mask-filter'
    
    #prompt suffix
    suffix += '_'+str(args.ptopk)+'_'+str(args.pt)+'_'+str(args.ntopk)+'_'+str(args.nt)

    #sd suffix    
    suffix += '_SD' + '_'+str(args.sd_weight)
    if args.pca: suffix+='_pca'
    if args.copca: suffix+='_copca'
    
    suffix+='_'+str(args.sd_layer_weight)
    suffix+='_'+str(args.start)+'_'+str(args.end)
    output_path = './outputs/' + '/' + args.outdir + '/' +args.data.split('/')[-1] + '/' + suffix 
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(output_path+'/log.txt','a+') as logger:
        logger.write("\n======>>>\n"+str(datetime.now())+'\n')
    
    
    #unseen id
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
    
    
    persam(args,  images_path,  output_path)
                   
      
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
  
       
def persam(args, images_path,  output_path):
    output_path = os.path.join(output_path)
    os.makedirs(output_path, exist_ok=True)
    print("\n------------> Segment " )

    target_feat_dict = {}
    class_iou = {}
    class_cnt = {}
    global_iou, global_cnt = 0, 0
    global_unique_list = []
    
    #enumrate class_id
    for class_id in range(0,255):
        # prepare ref_name
        if args.ref_txt!='x': # to do
            with open (os.path.join(args.data,args.ref_txt),'r') as f:
                lines = f.readlines()
                for line in lines:
                    x = line.split(' ') 
                    if x[0] == obj_name:
                        ref_name = x[1][:-1]
        elif args.ref_img!='x': ref_name = "ref_" + str(class_id)    
        else: 
            if args.ref_sed!='x': 
                fix_randseed(args.ref_sed)
                args.ref_idx = random.randint(0,len(os.listdir(os.path.join(images_path, obj_name)))-1)
            ref_name = sorted(os.listdir(os.path.join(images_path, obj_name)))[int(args.ref_idx)][:-4]
   
        # prepare ref_feat_path and ref_mask_path
        if args.ref_img!='x': ref_feat_path = os.path.join(images_path.replace("imgs",args.ref_img), ref_name + '.pth')
        else: ref_feat_path = os.path.join(images_path.replace("imgs",'sd_raw+dino_feat'), obj_name, ref_name + '.pth')
        
        if not os.path.exists(ref_feat_path): continue
        global_unique_list.append(class_id)
        
        ref_mask_path = ref_feat_path.replace('sd_raw+dino_feat','gts').replace('pth','png')
        
        # load ref_mask
        ref_mask = cv2.imread(ref_mask_path)
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
            if k == 's2': continue
            ref_feat2[k]= F.interpolate(v,size=(60,60),mode='nearest').squeeze().permute(1,2,0)
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
        target_feat_dict[class_id] = (target_feat1, target_feat2)
        class_iou[class_id], class_cnt[class_id] = 0, 0
        
    global_unique_list = list(set(global_unique_list))
        
    #load segment anything model
    print("======> Load SAM" )
    if args.sam_type == 'vit_b':
        sam_type, sam_ckpt = 'vit_b', '/data/tanglv/Ad-SAM/2023-9-7/Ad-Sam-Main/sam-continue-learning/pretrained_checkpoint/sam_vit_b_01ec64.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    
    # start testing
    print('======> Start Testing')
    test_images_path = images_path    
    test_files = sorted(os.listdir(test_images_path))

    if args.end==-1: args.end = len(os.listdir(test_images_path))-1
    args.end = min(args.end, len(os.listdir(test_images_path))-1)  
      
    for idx in tqdm(range(args.start,args.end+1)):
        # print(i)
        # raise NameError
        test_idx = test_files[idx]
        # Load test img 
        test_idx = test_idx[:-4]
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.resize(test_image,(1024,1024))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_img_torch = torch.from_numpy(test_image).permute(2,0,1).cuda()
        
        # Load test mask
        test_mask_path = test_images_path.replace('imgs','gts') + '/' + test_idx + '.png'
        test_masks = cv2.imread(test_mask_path,0)
        test_masks = cv2.resize(test_masks,(1024,1024),interpolation=cv2.INTER_NEAREST)
        # test_masks = cv2.cvtColor(test_masks, cv2.COLOR_BGR2GRAY)
        test_masks = torch.tensor(test_masks).cuda().unsqueeze(0).unsqueeze(0) # [1 1 H W]
                        
        # Load test feat
        test_feat_path = test_images_path.replace('imgs','sd_raw+dino_feat') + '/' + test_idx + '.pth'
        test_all_feat = torch.load(test_feat_path, map_location='cuda')
        sd_feat, dino_feat = test_all_feat['sd_feat'], test_all_feat['dino_feat']
        
        test_feat1 = dino_feat.reshape(60,60,768) 
        test_feat1  = test_feat1.permute(2,0,1)
        C, h, w = test_feat1.shape
        test_feat1 = test_feat1 / test_feat1.norm(dim=0, keepdim=True) # C h w
        test_feat1 = test_feat1.reshape(C, h * w) # C hw
         
        test_feat2 = {}
        for k,v in sd_feat.items():
            #[1 1280 15 15] [1 1280 30 30] [1 640 60 60]-> [60 60 1280] [60 60 1280] [60 60 640] 
            if k== 's2': continue 
            test_feat2[k]= F.interpolate(v,size=(60,60),mode='nearest').squeeze().permute(1,2,0)
            if args.pca: test_feat2[k] = pca(test_feat2[k].view(3600,-1)).view(60,60,-1)
  
            test_feat2[k]  = test_feat2[k].permute(2,0,1)
            C, h, w = test_feat2[k].shape
            test_feat2[k] = test_feat2[k] / test_feat2[k].norm(dim=0, keepdim=True) # C h w
            test_feat2[k] = test_feat2[k].reshape(C, h * w) # C hw
            

        unique_list = torch.unique(test_masks)
        unique_list = [x.item() for x in unique_list]
        
        # if unique_list[-1]==255: unique_list = unique_list[:-1]
        # if 'coco-stuff' in args.data: unique_list=unique_list[:-1]
        # if 'pcontext' in args.data: unique_list=unique_list[:-1]
        # if 'VOC2012' in args.data: unique_list=unique_list[1:-1]
    
        for class_id in unique_list:
            if class_id not in unseen_ids: continue
            #prepare matched target and test
            test_mask = torch.zeros_like(test_masks)
            test_mask[test_masks==class_id]=1
            target_feat1, target_feat2 = target_feat_dict[class_id][0], target_feat_dict[class_id][1]
            
            # Cosine similarity 1
            sim1 = target_feat1 @ test_feat1 # [1 C] @ [C hw]
            sim1 = sim1.reshape(1, 1, h, w)
        
            # Cosine similarity 2
            sim2 = 0
            for i,k in enumerate(test_feat2.keys()):
                # s5 s4 s3
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
            
            examples = []
            example = {}
            example['image'] = test_img_torch
            example['point_coords'] = torch.cat([p_coords,n_coords]).unsqueeze(0)
            example['point_labels'] = torch.cat([torch.ones(p_coords.shape[0]),torch.zeros(n_coords.shape[0])]).unsqueeze(0).cuda().to(torch.float32)                
            example['original_size'] = (1024, 1024)
            examples.append(example)
            
            with torch.no_grad():
                output = sam(examples, multimask_output=False)[0]
                masks, low_res_logits, iou_predictions = output['masks'],output['low_res_logits'],output['iou_predictions']
                
                # Cascaded Post-refinement-1
                best_idx = 0
                examples[0]['mask_inputs'] = low_res_logits[:,best_idx:best_idx+1,...]
                output = sam(examples, multimask_output=True)[0]
                masks, low_res_logits, iou_predictions = output['masks'],output['low_res_logits'],output['iou_predictions']
                
                # Cascaded Post-refinement-2
                best_idx = torch.argmax(iou_predictions[0]).item()
                examples[0]['mask_inputs'] = low_res_logits[:,best_idx:best_idx+1,...]

                y, x = torch.nonzero(masks[0,best_idx,...]).split(1,-1)
                if x.shape[0]==0: examples[0]['boxes'] = torch.tensor([[0,1023,0,1023]]).cuda().to(torch.float32)
                else: examples[0]['boxes'] = torch.tensor([[x.min(),y.min(),x.max(),y.max()]]).cuda().to(torch.float32)
                
                output = sam(examples, multimask_output=True)[0]
                masks, low_res_logits, iou_predictions = output['masks'],output['low_res_logits'],output['iou_predictions']
                
                # print(masks.shape, low_res_logits.shape, iou_predictions.shape)
                
            best_idx = torch.argmax(iou_predictions[0]).item()
            final_mask = masks[:,best_idx:best_idx+1,...]
            final_mask_np = final_mask.squeeze().cpu().numpy()
            
            #print(final_mask.shape, test_mask.shape, final_mask.max(), test_mask.max())
            iou = compute_iou(final_mask, test_mask)
            
            class_cnt[class_id] += 1
            class_iou[class_id] += iou
            global_cnt += 1
            global_iou += iou
            
            plt.figure(figsize=(10, 10))
            plt.imshow(test_image)
            show_mask(final_mask_np, plt.gca())    
            show_points(example['point_coords'][0].cpu().numpy(), example['point_labels'][0].cpu().numpy(), plt.gca())
            plt.title(f"Mask {best_idx}", fontsize=18)
            plt.axis('off')

            vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}_{str(class_id)}.jpg')
            if args.save:
                with open(vis_mask_output_path, 'wb') as outfile:
                    plt.savefig(outfile, format='jpg')
            
            mask_output_path = os.path.join(output_path, test_idx + '_' +str(class_id) +'.png')
            if args.save: cv2.imwrite(mask_output_path, final_mask_np.astype(np.uint8)*255)
        
        if global_cnt: 
            print("miou", global_iou/global_cnt)
            with open(output_path+'/log.txt','a') as logger:
                logger.write(str(idx)+ " miou:"+ str(global_iou/global_cnt) +'\n')
    
    with open(output_path+'/log.txt','a') as logger:    
        logger.write("\n\nAll INFO:\n")    
        if class_cnt[class_id]: logger.write( str(class_id)+' '+args.ref_img+' '+str(class_iou[class_id]/class_cnt[class_id])+'\n')
        logger.write('miou: '+str(global_iou/global_cnt)+'\n')
        

def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0) # [1 top_k] h
    topk_y = (topk_xy - topk_x * h)  #[1 top_k] w
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0) # [2 top_k] -> [tok_k 2]
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
        
    return topk_xy, topk_label, last_xy, last_label
    

if __name__ == "__main__":
    global sum_iou, sum_cnt, group_iou, group_cnt
    sum_iou  = 0
    sum_cnt = 0
    group_iou = 0
    group_cnt = 0
    
    main()