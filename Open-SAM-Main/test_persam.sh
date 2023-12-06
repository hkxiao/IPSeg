export CUDA_VISIBLE_DEVICES=1
python persam.py  \
    --data /data/tanglv/data/fss-te/perseg \
    --outdir persam_dino_kmeans/perseg \
    --sam_type vit_b \
    --sam_pth /data/tanglv/Ad-SAM/2023-9-7/Ad-Sam-Main/sam_continue_learning/train/work_dirs/diceloss_sam_iou_masktoken-tuning_b_adv@4/asam_epoch_9.pth
