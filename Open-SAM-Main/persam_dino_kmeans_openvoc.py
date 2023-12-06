import numpy as np
import torch
from torch.nn import functional as F
import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor
from process_feat import process_feat
from kmeans_pytorch import kmeans
import random

def binary_erosion(image, kernel):
    result = np.zeros_like(image)
    image_padded = np.pad(image, ((1,1), (1,1)), mode='constant')  # 在图像周围添加一圈0，以避免边界问题
    for i in range(1, image.shape[0] + 1):
        for j in range(1, image.shape[1] + 1):
            if np.all(image_padded[i-1:i+2, j-1:j+2] & kernel):
                result[i-1, j-1] = 1
    return result

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
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
    return iou / len(preds)

def get_arguments():
    #CUDA_VISIBLE_DEVICES=0 python persam_dino_kmeans.py
    parser = argparse.ArgumentParser()
    
    #ref setting
    parser.add_argument('--sed', default=0)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--ref_txt', action='store_true')
    parser.add_argument('--ref_idx', type=int, default=0)
    
    #base setting
    parser.add_argument('--data', type=str, default='/data/tanglv/data/fss-te/fold0')
    parser.add_argument('--outdir', type=str, default='persam_dino_kmeans/fold0')
    parser.add_argument('--one_shot', action='store_true')
    parser.add_argument('--matting', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--erosion', action="store_true")
    parser.add_argument('--prompt_filter', action="store_true")
    parser.add_argument('--mask_filter', action="store_true")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    
    args = parser.parse_args()
    fix_randseed(args.sed)
    return args


def main():
    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/imgs/'
    masks_path = args.data + '/gts/'
    output_path = './outputs/' + args.outdir

    
    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    persam(args, images_path,  output_path)
  
        
def get_point(correlation_map, topk=4): # N k H W
    correlation_max = torch.max(correlation_map, dim=1)[0] # N H W
    ranged_index = torch.argsort(torch.flatten(correlation_max, -2), 1, descending=True) #N HW
    coords = torch.stack([ranged_index[:,:32]%60,ranged_index[:,:32]/60],-1) #N 32 2
    centers = []
    for k in range(coords.shape[0]):
        center = kmeans(coords[k],K=topk, max_iters=20) #2 2
        centers.append(center)
    max_centers = torch.stack(centers,dim=0) #N k 2
    return max_centers        
        
    
def persam(args, test_images_path,  output_path):
    os.makedirs(output_path, exist_ok=True)
    print("\n------------> Segment " )
    
    # prepare target
    target_feat_dict = {}
    class_iou = {}
    class_cnt = {}
    global_iou, global_cnt = 0, 0
    
    print("======> Obtain  Target Feature" )
    for ref_idx in range(0,255):
        ref_name = str(ref_idx)
        if args.matting: ref_name = ref_name + '_matting'
        
        ref_feat_path = os.path.join(test_images_path.replace("imgs",'sd_raw+dino_feat'), 'ref_'+ref_name + '.pth')
        if not os.path.exists(ref_feat_path): continue
        
        if args.one_shot:
            ref_mask_path = ref_feat_path.replace('sd_raw+dino_feat','gts').replace('pth','png')
        else:
            ref_mask_path = ref_feat_path.replace('sd_raw+dino_feat','a2s').replace('pth','png')

        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)
        ref_mask = torch.tensor(ref_mask).cuda().unsqueeze(0).unsqueeze(0).to(torch.float32) # 1 1 H W

    
        # ref Image features encoding
        ref_all_feat = torch.load(ref_feat_path, map_location='cuda')
        sd_feat, dino_feat = ref_all_feat['sd_feat'], ref_all_feat['dino_feat'] # C 60 60 
        ref_feat = process_feat(sd_feat,dino_feat, ONLY_DINO=True)
        ref_feat = ref_feat.squeeze().permute(1,2,0)  # 60 60 C 
        
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear") #1 1 h w
        if args.one_shot: ref_mask[ref_mask!=0] = 1
        else: ref_mask = ref_mask / 255.0
        
        if args.erosion:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  
            ref_mask = -max_pool(-ref_mask)

        ref_mask = ref_mask.squeeze() # h w
            
        target_feat = ref_feat[ref_mask>0.5]  # N C   
        target_embedding = target_feat.mean(0).unsqueeze(0) # N C -> 1 C
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True) # 1 C
        target_feat_dict[ref_idx]=target_feat
        class_iou[ref_idx], class_cnt[ref_idx] = 0, 0
        
    
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

    predictor = SamPredictor(sam)

    print('======> Start Testing')
    for test_idx in tqdm(sorted(os.listdir(test_images_path))[args.start:args.end]):
        # Load test img and  feat
        test_idx = test_idx[:-4]
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.resize(test_image,(1024,1024))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(test_image) 
        
        # Image feature encoding
        test_feat_path = test_images_path.replace('imgs','sd_raw+dino_feat') + '/' + test_idx + '.pth'
        test_all_feat = torch.load(test_feat_path, map_location='cuda')
        sd_feat, dino_feat = test_all_feat['sd_feat'], test_all_feat['dino_feat']
        test_feat = process_feat(sd_feat,dino_feat, ONLY_DINO=True)
        test_feat = test_feat.squeeze() # C 60 60
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True) # C h w
        test_feat = test_feat.reshape(C, h * w) # C hw
        
        # Load test mask
        test_mask_path = test_images_path.replace('imgs','gts') + '/' + test_idx + '.png'
        test_masks = cv2.imread(test_mask_path)
        test_masks = cv2.resize(test_masks,(1024,1024),interpolation=cv2.INTER_NEAREST)
        test_masks = cv2.cvtColor(test_masks, cv2.COLOR_BGR2GRAY)
        test_masks = torch.tensor(test_masks).cuda().unsqueeze(0).unsqueeze(0) # [1 1 H W]
        
        unique_list = torch.unique(test_masks)
        unique_list = [x.item() for x in unique_list]
        if unique_list[-1]==255: unique_list = unique_list[:-1]
        
        for id in unique_list:
            if id == 255: continue
            #prepare matched target and test
            test_mask = torch.zeros_like(test_masks)
            test_mask[test_masks==id]=1
            target_feat = target_feat_dict[id]
            
            # Cosine similarity
            sim = target_feat @ test_feat # [1 C] @ [C hw]
            sim = sim.reshape(1, 1, h, w)
            
            # Positive-negative location prior      
            p_coords, n_coords = get_point(sim,topk=4), get_point(1-sim,topk=4) # [1 4 2]
            p_coords, n_coords = p_coords.squeeze(),  n_coords.squeeze() # [4 2]
            
            #Clear prompt
            tmp_p_coords, tmp_n_coords = torch.empty(0,2).cuda(), torch.empty(0,2).cuda()
            for p_coord,n_coord in zip(p_coords, n_coords):            
                if not torch.isnan(p_coord).any().item(): tmp_p_coords = torch.cat([tmp_p_coords, p_coord.unsqueeze(0)])
                if not torch.isnan(n_coord).any().item(): tmp_n_coords = torch.cat([tmp_n_coords, n_coord.unsqueeze(0)])
            p_coords=tmp_p_coords 
            n_coords=tmp_n_coords 
            
            p_coords, n_coords = p_coords*1024/60+1024/120, n_coords*1024/60+1024/120
            topk_xy = torch.cat([p_coords,n_coords],0).cpu().numpy()
            topk_label = np.concatenate([np.ones(p_coords.shape[0]),np.zeros(n_coords.shape[0])], 0)
            
            # First-step prediction
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy, 
                point_labels=topk_label, 
                multimask_output=False
            )
            best_idx = 0

            # Cascaded Post-refinement-1
            masks, scores, logits, _ = predictor.predict(
                        point_coords=topk_xy,
                        point_labels=topk_label,
                        mask_input=logits[best_idx: best_idx + 1, :, :], 
                        multimask_output=True)
            best_idx = np.argmax(scores)

            # Cascaded Post-refinement-2
            y, x = np.nonzero(masks[best_idx])
            
            if x.shape[0]==0:
                x_min = 0
                x_max = 1023
                y_min = 0
                y_max = 1023
            else:
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
            
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box[None, :],
                mask_input=logits[best_idx: best_idx + 1, :, :], 
                multimask_output=True)
            best_idx = np.argmax(scores)

            # Save masks
            plt.figure(figsize=(10, 10))
            plt.imshow(test_image)
            show_mask(masks[best_idx], plt.gca())
            show_points(topk_xy, topk_label, plt.gca())
            plt.title(f"Mask {best_idx}", fontsize=18)
            plt.axis('off')
            vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}_{str(id)}.jpg')
            if args.save: 
                print("save")
                with open(vis_mask_output_path, 'wb') as outfile:
                    plt.savefig(outfile, format='jpg')

            final_mask = masks[best_idx]
            
            iou = compute_iou(torch.tensor(final_mask).unsqueeze(0).unsqueeze(0).cuda(), test_mask)
            #print(iou)
            class_cnt[id] += 1
            class_iou[id] += iou
            global_cnt += 1
            global_iou += iou
            
            mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            mask_colors[final_mask, :] = np.array([[0, 0, 128]])   
            mask_output_path = os.path.join(output_path, test_idx+'_'+str(id)+ '.png')
            if args.save: cv2.imwrite(mask_output_path, mask_colors)

        print("Now IOU", global_iou/global_cnt)
    
    for id in target_feat_dict.keys():
        print(str(id)+" IOU", class_iou[id]/class_cnt[id])   
    

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
