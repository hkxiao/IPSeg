export CUDA_VISIBLE_DEVICES=0

for data in fold0 fss;
do
    python  opensam_fss_gt.py --sam_type vit_h  \
        --data /data/tanglv/data/fss-te/${data} \
        --ref_txt ref_composed.txt \
        --erosion \
        --ptopk=32 \
        --pt=4 \
        --ntopk=32 \
        --nt=4 \
        --vit_weight 1.0 \
        --vit_size base \
        --vit_type dinov2 \
        --sd_weight 0.0 \
        --sd_layer_weight=0.3,0.2,0.1 \
        --visualize 
done
wait