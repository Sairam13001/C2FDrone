#!/usr/bin/env bash
# swin_b_p4w7

set -x
T=`date +%m%d%H%M`

EXP_DIR=checkpoints/
mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --eval \
    --backbone swin_b_p4w7 \
    --epochs 15 \
    --lr_drop_epochs 5 6 \
    --batch_size 1 \
    --num_workers 1 \
    --num_frames 1 \
    --num_feature_levels 1\
    --num_queries 100 \
    --resume expms/FL_640_ablations/swinbp4w7_fpn-style-da-plus-fpn_OE-losses_init-dec-queries_fpn-feats-to-dabdetr_lr8e-5_bs2_ep15_gpus1_lrdrop5-6_Dis-T_TDL-T_Shuf-T/checkpoint0014.pth \
    --with_box_refine \
    --hidden_dim 256 \
    --dataset_file 'vid_single' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.eval_on_testset_ep15.$T.txt
