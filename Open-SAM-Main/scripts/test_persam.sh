export CUDA_VISIBLE_DEVICES=1
python persam.py  \
    --data /data/tanglv/data/fss-te/perseg \
    --outdir persam/perseg \
    --sam_type vit_h \
    --sam_pth sam_vit_h_4b8939.pth