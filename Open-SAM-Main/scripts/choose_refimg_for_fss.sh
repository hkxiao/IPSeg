export CUDA_LAUNCH_BLOCKING=True
export CUDA_VISIBLE_DEVICES=0
python best_refimg_fss.py  \
    --erosion \
    --sd_weight=0.1 \
    --sd_layer_weight=0.3,0.2,0.1 \
    --datasets fss


export CUDA_VISIBLE_DEVICES=0
python  opensam_fss.py --sam_type vit_h  \
    --data /data/tanglv/data/fss-te/fss \
    --ref_txt=match_point_ref0.txt \
    --erosion \
    --ptopk=32 \
    --pt=4 \
    --ntopk=32 \
    --nt=4 \
    --sd_weight=0.1 \
    --sd_layer_weight=0.3,0.2,0.1 
    --save




