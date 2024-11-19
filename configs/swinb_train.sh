#!/usr/bin/env bash
#swin_b_p4w7

set -x
T=`date +%m%d%H%M`
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone ${backbone} \
    --epochs ${epochs} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --num_frames ${num_frames} \
    --num_feature_levels 1\
    --num_queries ${num_queries} \
    --start_epoch ${start_epoch} \
    --dilation \
    --hidden_dim 256 \
    --with_box_refine \
    --lr_drop_epochs 5 6 \
    --coco_pretrain \
    --resume expms/pre-trained-model/swinb_checkpoint0048.pth \
    --gap 1 \
    --output_dir ${EXP_DIR} \
    --dataset_file 'vid_single' \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T

