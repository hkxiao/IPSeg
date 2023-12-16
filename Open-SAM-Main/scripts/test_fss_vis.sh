export CUDA_VISIBLE_DEVICES=0
python  opensam_fss.py --sam_type vit_h  \
    --data /data/tanglv/data/fss-te/fold0 \
    --ref_txt=ref_composed.txt \
    --erosion \
    --ptopk=32 \
    --pt=4 \
    --ntopk=32 \
    --nt=4 \
    --sd_weight=0.1 \
    --sd_layer_weight=0.3,0.2,0.1 \
    --visualize