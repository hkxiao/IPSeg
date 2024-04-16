from transformers import AutoProcessor, CLIPModel
from PIL import Image
import requests
import tqdm
import os
from pathlib import Path
import torch
from tqdm import tqdm

datas = ['perseg']
root = '/data/tanglv/data/fss-te'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
# model.config.image_size = 224

for data in tqdm(datas):

    img_dir = os.path.join(root, data, 'imgs')
    mae_dir = os.path.join(root, data, 'clip_feats')
    Path(mae_dir).mkdir(exist_ok=True, parents=True)
    
    for group in tqdm(os.listdir(img_dir)):
        if group.startswith('.'): continue

        mae_group_dir = os.path.join(root, data, 'clip_feats', group)
        group_dir = os.path.join(root, data, 'imgs', group)
        Path(mae_group_dir).mkdir(exist_ok=True, parents=True)
            
        for file in  tqdm(os.listdir(group_dir)):
            if not file.endswith('jpg'): continue
            
            image = Image.open(os.path.join(group_dir, file))

            inputs = processor(images=image, return_tensors="pt")
            image_features = model.get_image_features(**inputs,output_hidden_states=True)[0]
            #print(model.config.mask_ratio,model.config.image_size)
            print(image_features.shape)
            torch.save(image_features, os.path.join(mae_group_dir, file.replace('jpg','pth')))