import numpy as np
import torch
from torch.nn import functional as F
import clip

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from sklearn.decomposition import PCA
 
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor


def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='/data/tanglv/data/perseg')
    parser.add_argument('--outdir', type=str, default='persam')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_l')
    
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            opensam(args, obj_name, images_path, masks_path, output_path)


def opensam(args, obj_name, images_path, masks_path, output_path):

    print("\n------------> Segment " + obj_name)
    
    # Path preparation
    test_images_path = os.path.join(images_path, obj_name)

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    
    print("======> Load CLIP" )
    model, preprocess = clip.load("ViT-B/32",device='cuda')  
    
    print("======> Load SAM" )
    if args.sam_type == 'vit_b':
        sam_type, sam_ckpt = 'vit_b', '/data/tanglv/xhk/Ad-Sam-Main/sam_continue_learning/pretrained_checkpoint/sam_vit_b_01ec64.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_l':
        sam_type, sam_ckpt = 'vit_l', 'sam_vit_l_0b3195.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()


    predictor = SamPredictor(sam)

    print("======> Obtain Text Prior" )
    # Text features encoding
    text = clip.tokenize([obj_name]).cuda()
    with torch.no_grad():
        text_feat = model.encode_text(text).to(torch.float32)
        text_feat = text_feat / text_feat.norm(-1, keepdim=True)

    print('======> Start Testing')
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):
        # Load test image
        test_idx = '%02d' % test_idx

        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Image feature encoding
        predictor.set_image(test_image)
        
        test_feat = predictor.interm_embeddings[-1].squeeze()
        h,w,C = test_feat.shape
        test_feat =  test_feat.reshape(-1,C)
        
        pca = PCA(n_components=512)
        test_feat = pca.fit_transform(test_feat.cpu())
        test_feat = torch.from_numpy(test_feat).to(torch.float32).cuda()  # h*w 512
    
        test_feat = test_feat / test_feat.norm(dim=-1, keepdim=True) # h*w 512
        test_feat = test_feat.permute(1,0)# 512 h*w
        sim = text_feat @ test_feat # [1 512]@ [512 h*w] = [1 h*w]

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()

        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
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
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
           # target_embedding=target_embedding  # Target-semantic Prompting
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
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, test_idx + '.png')
        cv2.imwrite(mask_output_path, mask_colors)


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
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
