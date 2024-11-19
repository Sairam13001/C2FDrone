#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=expms/swin_s/NPS_Drones/swin_s_p4w7_ep7_bs4_gpus2_lr2e-5_lrbb2e-5_lr_drop_epochs_5-6_aug-RHFlip-Resize_640x640-colorjitter-noshuf
mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_s_p4w7 \
    --epochs 7 \
    --lr_drop_epochs 5 6 \
    --num_feature_levels 1\
    --num_queries 300 \
    --dilation \
    --batch_size 4 \
    --hidden_dim 256 \
    --num_workers 1 \
    --with_box_refine \
    --resume /workspace/expms/pre-trained-model/swins_checkpoint0049.pth \
    --coco_pretrain \
    --dataset_file 'vid_single' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
