from transformers import AutoImageProcessor, ViTMAEModel
from PIL import Image
import requests
import tqdm
import os
from pathlib import Path
import torch
from tqdm import tqdm

datas = ['fold0','fold1','fold2','fold3','perseg','fss']
root = '/data/tanglv/data/fss-te'
processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
model = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
model.config.mask_ratio = 0
# model.config.image_size = 224

for data in tqdm(datas):

    img_dir = os.path.join(root, data, 'imgs')
    mae_dir = os.path.join(root, data, 'mae_feats')
    Path(mae_dir).mkdir(exist_ok=True, parents=True)
    
    for group in tqdm(os.listdir(img_dir)):
        if group.startswith('.'): continue
        
        mae_group_dir = os.path.join(root, data, 'mae_feats', group)
        group_dir = os.path.join(root, data, 'imgs', group)
        Path(mae_group_dir).mkdir(exist_ok=True, parents=True)
            
        for file in  tqdm(os.listdir(group_dir)):
            if not file.endswith('jpg'): continue
            
            image = Image.open(os.path.join(group_dir, file)).convert('RGB')

            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs,output_hidden_states=True)

            hidden_state = outputs.last_hidden_state
            #print(model.config.mask_ratio,model.config.image_size)
            print(hidden_state.shape)
            torch.save(hidden_state, os.path.join(mae_group_dir, file.replace('jpg','pth')))