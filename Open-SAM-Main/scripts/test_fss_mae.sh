export CUDA_VISIBLE_DEVICES=4

for data in perseg;
do
    python  opensam_fss.py --sam_type vit_h  \
        --data /data/tanglv/data/fss-te/${data} \
        --ref_txt=ref_composed.txt \
        --erosion \
        --ptopk=32 \
        --pt=4 \
        --ntopk=32 \
        --nt=4 \
        --vit_weight 1.0 \
        --vit_size base \
        --vit_type clip \
        --sd_weight 0.1 \
        --sd_layer_weight=0.3,0.2,0.1 
done
wait