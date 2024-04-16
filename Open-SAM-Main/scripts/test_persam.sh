export CUDA_VISIBLE_DEVICES=3
python persam.py  \
    --data /data/tanglv/data/fss-te/fold1 \
    --outdir persam/fold1 \
    --sam_type vit_h \
    --sam_pth pretrained/sam_vit_h_4b8939.pth
