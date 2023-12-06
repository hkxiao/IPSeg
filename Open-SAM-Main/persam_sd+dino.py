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

def compute_iou(preds, target):
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
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='/data/tanglv/data/perseg')
    parser.add_argument('--outdir', type=str, default='persam')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_b')
    
    args = parser.parse_args()
    return args

sum_iou  =0
sum_cnt = 0
group_iou = 0
group_cnt = 0

def main():

    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/imgs'
    masks_path = args.data + '/gts/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    global sum_iou, sum_cnt, group_iou, group_cnt
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            group_iou,group_cnt = 0,0
            persam(args, obj_name, images_path, masks_path, output_path)
            sum_iou += group_iou
            sum_cnt += group_cnt
            print(obj_name,"miou",group_iou/group_cnt)
                    
    print("ALL miou",sum_iou/sum_cnt)    
        
def persam(args, obj_name, images_path, masks_path, output_path):

    print("\n------------> Segment " + obj_name)
    
    # Path preparation
    ref_feat_path = os.path.join(images_path.replace("imgs",'sd_cp+dino_feat'), obj_name, args.ref_idx + '.pth')
    ref_mask_path = os.path.join('/data/tanglv/Open-SAM/2023-10-3/A2S-v2/result/cornet/resnet/base/c_perseg/final', obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)
    ref_mask = torch.tensor(ref_mask).cuda().unsqueeze(0).unsqueeze(0).to(torch.float32) # 1 1 H W
    
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

    print("======> Obtain Location Prior" )
    
    # ref Image features encoding
    ref_all_feat = torch.load(ref_feat_path, map_location='cuda')
    sd_feat, dino_feat = ref_all_feat['sd_feat'], ref_all_feat['dino_feat'] # C 60 60 
    ref_feat = process_feat(sd_feat,dino_feat)
    ref_feat = ref_feat.squeeze().permute(1,2,0)  # 60 60 C 

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear") 
    ref_mask = ref_mask.squeeze() # 1 1 h w

    target_feat = ref_feat[ref_mask >125.0]  # N C
    #print(target_feat.shape,ref_feat.shape,ref_mask.shape)  #[N C] [60 60 C]  [C C]
    
    target_embedding = target_feat.mean(0).unsqueeze(0) # N C -> 1 C
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True) # 1 C
    target_embedding = target_embedding.unsqueeze(0) # 1 1 C

    print('======> Start Testing')
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):
    
        # Load test img and  feat
        test_idx = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.resize(test_image,(1024,1024))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(test_image) 
        
        # Load test mask
        test_mask_path = test_images_path.replace('imgs','gts') + '/' + test_idx + '.png'
        test_mask = cv2.imread(test_mask_path)
        test_mask = cv2.resize(test_mask,(1024,1024))
        test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
        test_mask = (test_mask>0).astype(np.float32)
        test_mask = torch.tensor(test_mask).cuda().unsqueeze(0).unsqueeze(0) # [1 1 H W]
        
        # Image feature encoding
        test_feat_path = test_images_path.replace('imgs','sd_cp+dino_feat') + '/' + test_idx + '.pth'
        test_all_feat = torch.load(test_feat_path, map_location='cuda')
        sd_feat, dino_feat = test_all_feat['sd_feat'], test_all_feat['dino_feat']
        test_feat = process_feat(sd_feat,dino_feat)
        test_feat = test_feat.squeeze() # C 60 60

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True) # C h w
        test_feat = test_feat.reshape(C, h * w) # C hw
        sim = target_feat @ test_feat # [1 C] @ [C hw]
        
        sim = sim.reshape(1, 1, h, w)
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze() # [H W]

        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=5)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

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
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')

        final_mask = masks[best_idx]
        
        global group_cnt, group_iou
        group_cnt = group_cnt + 1
        group_iou += compute_iou(torch.tensor(final_mask).unsqueeze(0).unsqueeze(0).cuda(), test_mask)
        
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, test_idx + '.png')
        cv2.imwrite(mask_output_path, mask_colors)


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0) # [1 top_k] h
    topk_y = (topk_xy - topk_x * h)  #[1 top_k] w
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0) # [2 top_k] -> [tok_k 2]
    # print(topk_xy.shape)
    # raise NameError
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
    main()
