export CUDA_VISIBLE_DEVICES=0
python opensam_voc.py --sam_type vit_h  \
    --data /data/tanglv/data/openvoc-te/pcontext \
    --outdir  openvoc\
    --ref_img refimg0 \
    --erosion \
    --sd_weight=0.1 \
    --sd_layer_weight=0.3,0.2,0.1 \
    --start=0 \
    --save 