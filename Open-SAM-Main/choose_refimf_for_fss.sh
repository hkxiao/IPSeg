export CUDA_LAUNCH_BLOCKING=True
export CUDA_VISIBLE_DEVICES=1
python best_refimg_fss.py  \
    --erosion \
    --sd_weight=0.1 \
    --sd_layer_weight=0.3,0.2,0.1 \
    --datasets fold2


