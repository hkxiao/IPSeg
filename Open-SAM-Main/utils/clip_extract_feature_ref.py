from transformers import AutoProcessor, CLIPModel
from PIL import Image
import requests
import tqdm
import os
from pathlib import Path
import torch
from tqdm import tqdm

datas = ['fold0','fold1','fold2','fold3','perseg','fss']
root = '/data/tanglv/data/fss-te'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
# model.config.image_size = 224

for data in tqdm(datas):
    for group in tqdm(['refimg0', 'refimg1', 'refimg2']):        
        group_dir = os.path.join(root, data, group)
            
        for file in  tqdm(os.listdir(group_dir)):
            if not file.endswith('jpg') or 'matting' in file: continue
            
            image = Image.open(os.path.join(group_dir, file)).convert('RGB')

            inputs = processor(images=image, return_tensors="pt")
            image_features = model.get_image_features(**inputs,output_hidden_states=True)[0]
            #print(model.config.mask_ratio,model.config.image_size)
            print(image_features.shape)
            #print(os.path.join(group, file.replace('.jpg','_mae.pth')))
            torch.save(image_features, os.path.join(group_dir, file.replace('.jpg','_clip.pth')))