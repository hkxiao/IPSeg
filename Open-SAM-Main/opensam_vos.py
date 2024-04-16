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

from show import *
from segment_anything import sam_model_registry, SamPredictor
from process_feat import process_feat
from kmeans_pytorch import kmeans
import random
from pathlib import Path
from utils.evaluation import Evaluator
from utils.logger import AverageMeter, Logger

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
    
def db_eval_iou(annotation,segmentation):

    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)

def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);

    return F

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap

def get_arguments():
    #CUDA_VISIBLE_DEVICES=0 python persam_dino_kmeans.py
    parser = argparse.ArgumentParser()
    
    #ref setting
    parser.add_argument('--ref_txt', default='x')
    parser.add_argument('--ref_img', default='x')
    parser.add_argument('--ref_sed', default='x')
    parser.add_argument('--ref_idx', default='x')
    
    #vit setting 
    parser.add_argument('--vit_type', type=str, default='dinov2')
    parser.add_argument('--vit_size', type=str, default='vit_b')
    parser.add_argument('--vit_weight', type=float, default=0.)

    #sd setting
    parser.add_argument('--sd_weight', type=float, default=0.)
    parser.add_argument('--sd_layer_weight', type=str, default="1,1,1")
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--copca', action='store_true')
    
    #trick setting
    parser.add_argument('--oneshot', action='store_true')
    parser.add_argument('--zeroshot', action='store_true')

    parser.add_argument('--matting', action='store_true')
    parser.add_argument('--erosion', action="store_true")
    parser.add_argument('--prompt_filter', action="store_true")
    parser.add_argument('--mask_filter', action="store_true")
    
    # prompt setting
    parser.add_argument('--ptopk', type=int, default=32)
    parser.add_argument('--pt', type=int, default=4)
    parser.add_argument('--ntopk', type=int, default=32)
    parser.add_argument('--nt', type=int, default=4)
    
    # sam setting
    parser.add_argument('--sam_type', type=str, default='vit_h')
    
    #base setting
    parser.add_argument('--data', type=str, default='/data/tanglv/data/fss-te/fold0')
    parser.add_argument('--outdir', type=str, default='fss-te')
    parser.add_argument('--visualize', action='store_true')
        
    args = parser.parse_args()
    
    args.sd_layer_weight = args.sd_layer_weight.split(',')
    args.sd_layer_weight = [float(x) for x in args.sd_layer_weight]
    return args


def main():
    args = get_arguments()
    print("Args:", args)
        
    # prepare path
    images_path = args.data + '/imgs/'
    
    #ref suffix
    suffix = args.ref_txt+'_'+args.ref_img + '_' + args.ref_sed + '_' + args.ref_idx

    #vit suffix    
    suffix += '_VIT_' + str(args.vit_type) +   '_' + str(args.vit_size) + '_' + str(args.vit_weight)

    #sd suffix    
    suffix += '_SD' + '_'+str(args.sd_weight)
    if args.pca: suffix+='_pca'
    if args.copca: suffix+='_copca'
    suffix+='_'+str(args.sd_layer_weight)
    
    #trick suffix
    if args.erosion: suffix += '_erosion'
    if args.oneshot: suffix += '_oneshot'
    if args.zeroshot: suffix += '_zeroshot'
    if args.prompt_filter: suffix += '_prompt-filter'
    if args.mask_filter: suffix += '_mask-filter'
    
    #prompt suffix
    suffix += '_'+str(args.ptopk)+'_'+str(args.pt)+'_'+str(args.ntopk)+'_'+str(args.nt)

    output_path = './outputs/' + '/' + args.outdir + '/' +args.data.split('/')[-1] + '/' + suffix 
    Path(output_path).mkdir(parents=True, exist_ok=True)
    logger = open(output_path+'/log.txt','w') 
    
    
    #load segment anything model
    print("======> Load SAM" )
    if args.sam_type == 'vit_b':
        sam_type, sam_ckpt = 'vit_b', '/data/tanglv/Ad-SAM/2023-9-7/Ad-Sam-Main/sam-continue-learning/pretrained_checkpoint/sam_vit_b_01ec64.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'pretrained/sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    
    global sum_iou, sum_f, sum_cnt, group_iou, group_f, group_cnt
    
    for obj_name in tqdm(sorted(os.listdir(images_path))):
        #print('fancy_boot')
        #obj_name = 'fancy_boot'
        if ".DS" not in obj_name:
            group_iou, group_f, group_cnt = 0, 0, 0
            opensam(sam ,args, obj_name, images_path,  output_path, logger)
            sum_iou += group_iou / group_cnt
            sum_f += group_f / group_cnt
            sum_cnt += 1
            print(obj_name,"miou",group_iou/group_cnt,'f',group_f/group_cnt)    
            logger.write(' '+str(group_iou/group_cnt)+' '+str(group_f/group_cnt)+'\n')    
            print("Now ALL miou",sum_iou/sum_cnt)
            print("Now ALL mf",sum_f/sum_cnt)

            # break
    
    logger.write("All miou: "+str(sum_iou/sum_cnt)+'\n')    
    logger.write("All f: "+str(sum_f/sum_cnt)+'\n')    
    logger.write("All j&f: "+str((sum_f+sum_iou)/sum_cnt/2)+'\n')    

    logger.close()
    
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
        
def opensam(sam, args, obj_name, images_path,  output_path, logger):
    print("\n------------> Segment " + obj_name)
    
    # prepare ref_name
    if args.ref_txt!='x': 
        with open (os.path.join(args.data,args.ref_txt),'r') as f:
            lines = f.readlines()
            for line in lines:
                # print(line)
                x = line.split(' ') 
                if x[0] == obj_name:
                    ref_name = x[1][:-1]
    elif args.ref_img!='x': ref_name = "ref_" + str(obj_name)    
    else: 
        if args.ref_sed!='x': 
            fix_randseed(args.ref_sed)
            args.ref_idx = random.randint(0,len(os.listdir(os.path.join(images_path, obj_name)))-1)
        ref_name = sorted(os.listdir(os.path.join(images_path, obj_name)))[int(args.ref_idx)][:-4]
    print("\n------------> ref_name:", ref_name) 
    
    if args.ref_img!='x': logger.write(obj_name+' '+args.ref_img+'/'+ref_name)  
    else: logger.write(obj_name+' '+ref_name)    

    # prepare ref_feat_path and ref_mask_path
    if args.ref_img!='x': ref_feat_path = os.path.join(images_path.replace("imgs",args.ref_img), ref_name + '.pth')
    elif '/' in ref_name: ref_feat_path = os.path.join(images_path.replace("imgs/", ref_name) +  '.pth')
    else: ref_feat_path = os.path.join(images_path.replace("imgs",'sd_raw+dino_feat'), obj_name, ref_name + '.pth')
    print('ref_feat_path: ', ref_feat_path)
    
    if args.oneshot:
        print("one shot setting")
        ref_mask_path = ref_feat_path.replace('sd_raw+dino_feat','gts').replace('pth','png')
    else:
        ref_mask_path = ref_feat_path.replace('sd_raw+dino_feat','a2s').replace('pth','png')

        # if "sd_raw+dino_feat" in ref_feat_path:
        #     ref_mask_path = ref_feat_path.replace('sd_raw+dino_feat','a2s').replace('pth','png')
        # else:
        #     ref_mask_path = ref_feat_path.replace('.pth','_tsdn.png')
            
    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

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
    
    if args.zeroshot:
        ref_mask = torch.zeros_like(ref_mask)
    
    # load ref_feat
    ref_all_feat = torch.load(ref_feat_path, map_location='cuda')
    sd_feat, dino_feat = ref_all_feat['sd_feat'], ref_all_feat['dino_feat'] #  [1 1 3600 768]  
    ref_feat1 = dino_feat.reshape(60,60,768)  
    if args.vit_type == 'mae':
        if 'sd_raw+dino_feat' in ref_feat_path:
            ref_feat1 = torch.load(ref_feat_path.replace('sd_raw+dino_feat','mae_feats')).cuda()
        else:
            ref_feat1 = torch.load(ref_feat_path.replace('.pth','_mae.pth')).cuda()
        ref_feat1 = ref_feat1[:,1:,:].permute(0,2,1)
        ref_feat1 = F.interpolate(ref_feat1.reshape(1,768,14,14),(60,60),mode='bilinear',align_corners=False)
        ref_feat1 = ref_feat1.squeeze().permute(1,2,0)
    if args.vit_type == 'clip':
        if 'sd_raw+dino_feat' in ref_feat_path:
            ref_feat1 = torch.load(ref_feat_path.replace('sd_raw+dino_feat','clip_feats')).cuda()
        else:
            ref_feat1 = torch.load(ref_feat_path.replace('.pth','_clip.pth')).cuda()
        print(ref_feat1.shape)
        ref_feat1 = ref_feat1[:,1:,:].permute(0,2,1)
        ref_feat1 = F.interpolate(ref_feat1.reshape(1,768,14,14),(60,60),mode='bilinear',align_corners=False)
        ref_feat1 = ref_feat1.squeeze().permute(1,2,0)    
    
    ref_feat2 = {}
    for k,v in sd_feat.items():
        #[1 1280 15 15] [1 1280 30 30] [1 640 60 60]-> [60 60 1280] [60 60 1280] [60 60 640] 
        if k == 's2': continue
        ref_feat2[k]= F.interpolate(v,size=(60,60),mode='nearest').squeeze().permute(1,2,0)
        if args.pca: ref_feat2[k] = pca(ref_feat2[k].view(3600,-1)).view(60,60,-1)
    
    # get target feat1
    target_feat1 = ref_feat1[ref_mask>0.5]  # N C   
    target_feat1 = target_feat1.mean(0).unsqueeze(0) # N C -> 1 C
    target_feat1 = target_feat1 / target_feat1.norm(dim=-1, keepdim=True) # 1 C
    
    # get target feat2  
    target_feat2 = {}
    for k,v in ref_feat2.items():
        target_feat2[k] = v[ref_mask>0.5] # N C
        target_feat2[k] = target_feat2[k].mean(0).unsqueeze(0) # N C -> 1 C
        target_feat2[k] = target_feat2[k] / target_feat2[k].norm(dim=-1, keepdim=True) # 1 C
    
    # start testing
    print('======> Start Testing',obj_name)
    test_images_path = os.path.join(images_path, obj_name)    
    for test_idx in tqdm(sorted(os.listdir(test_images_path))):
        #print(test_idx)
        if not test_idx.endswith('jpg'): continue
        # Load test img 
        test_idx = test_idx[:-4]
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.resize(test_image,(1024,1024))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_img_torch = torch.from_numpy(test_image).permute(2,0,1).cuda()
        
        # Load test mask
        test_mask_path = test_images_path.replace('imgs','gts') + '/' + test_idx + '.png'
        test_mask = cv2.imread(test_mask_path)
        test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
        test_mask = cv2.resize(test_mask,(1024,1024))
        test_mask = torch.tensor(test_mask).cuda().unsqueeze(0).unsqueeze(0) # [1 1 H W]
        test_mask[test_mask>0] =1
        
        # Load test feat
        test_feat_path = test_images_path.replace('imgs','sd_raw+dino_feat') + '/' + test_idx + '.pth'
        test_all_feat = torch.load(test_feat_path, map_location='cuda')
        sd_feat, dino_feat = test_all_feat['sd_feat'], test_all_feat['dino_feat']
        
        test_feat1 = dino_feat.reshape(60,60,768)  
        if args.vit_type == 'mae':
            test_feat1 = torch.load(test_feat_path.replace('sd_raw+dino_feat','mae_feats')).cuda()
            test_feat1 = test_feat1[:,1:,:].permute(0,2,1)
            test_feat1 = F.interpolate(test_feat1.reshape(1,768,14,14),(60,60),mode='bilinear',align_corners=False)
            test_feat1 = test_feat1.squeeze().permute(1,2,0)
        if args.vit_type == 'clip':
            test_feat1 = torch.load(test_feat_path.replace('sd_raw+dino_feat','clip_feats')).cuda()
            test_feat1 = test_feat1[:,1:,:].permute(0,2,1)
            test_feat1 = F.interpolate(test_feat1.reshape(1,768,14,14),(60,60),mode='bilinear',align_corners=False)
            test_feat1 = test_feat1.squeeze().permute(1,2,0)
      
        test_feat2 = {}
        for k,v in sd_feat.items():
            #[1 1280 15 15] [1 1280 30 30] [1 640 60 60]-> [60 60 1280] [60 60 1280] [60 60 640] 
            if k == 's2': continue
            test_feat2[k]= F.interpolate(v,size=(60,60),mode='nearest').squeeze().permute(1,2,0)
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
        sim = sim1 * args.vit_weight + sim2 * args.sd_weight
        
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
        
        # sam input
        examples = []
        example = {}
        example['image'] = test_img_torch
        example['point_coords'] = torch.cat([p_coords,n_coords]).unsqueeze(0)
        example['point_labels'] = torch.cat([torch.ones(p_coords.shape[0]),torch.zeros(n_coords.shape[0])]).unsqueeze(0).cuda().to(torch.float32)                
        example['original_size'] = (1024, 1024)
        examples.append(example)
        
        # sam process
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
                    
        best_idx = torch.argmax(iou_predictions[0]).item()
        final_mask = masks[:,best_idx:best_idx+1,...]
        final_mask_np = final_mask.squeeze().cpu().numpy()
        
        global group_cnt, group_iou, group_f
        group_cnt = group_cnt + 1
        group_iou += db_eval_iou(final_mask.squeeze().detach().cpu().numpy(), test_mask.squeeze().detach().cpu().numpy())
        group_f += db_eval_boundary(final_mask.squeeze().detach().cpu().numpy(), test_mask.squeeze().detach().cpu().numpy())
               
        #visualize
        if args.visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(test_image)
            show_mask(final_mask_np, plt.gca())    
            show_points(example['point_coords'][0].cpu().numpy(), example['point_labels'][0].cpu().numpy(), plt.gca())
            plt.title(f"Mask {best_idx}", fontsize=18)
            plt.axis('off')
            vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
            
            with open(vis_mask_output_path, 'wb') as outfile:
                plt.savefig(outfile, format='jpg')
        
            mask_output_path = os.path.join(output_path, test_idx + '.jpg')
            cv2.imwrite(mask_output_path, final_mask_np.astype(np.uint8)*255)
        
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
    sum_f = 0
    group_iou = 0
    group_cnt = 0
    
    Evaluator.initialize()
    main()