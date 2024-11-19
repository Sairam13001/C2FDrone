
# expms/NPS_Drones_val_640_inpainted/swinbp4w7_fpn-style-DA-instance-aware-ce-loss-wt1-inverted-attn-maps-sigmoid-dice-loss-wt3_DA-feats-given-to-dabdedetr_lr8e-5_numframes3_ep12_lrdrop5-6_with-distributed_trained_on_inpainted_train_frames
# expms/FL_Drones_hs_val_from_june_19/swinbp4w7_fpn-style-DA-instance-aware-ce-loss-wt2-inverted-attn-maps-sigmoid-dice-loss-wt3_decoder-queries-init-by-attn-maps-80-20_DA-feats-given-to-dabdedetr_lr8e-5_numframes3_ep12_lrdrop5-6_with-distributed_trained_on_inpainted_train_frames
#expms/FL_Drones_val_from_jun21/swinbp4w7_fpn-style-DA-plus-fpn_i-aware-ce-loss-wt1-inv-attn-maps-sig-dice-loss-wt3_dec-quer-init-using-attn-maps-80-20_dec-query-loss-wt1-after-2eps_dec-query-WH-init-bw-0-0.1-plus-loss-wt1_DA-feats-given-to-dabdetr_lr8e-5_nfram3_ep12_gpus2_lrdrop5-6-distrib

EXP_DIR=/mnt/2C0EACF20EACB5EC/Drone-to-Drone-Detection_1/expms/Final_model
backbone=swin_b_p4w7; epochs=15; batch_size=1; num_workers=0; num_frames=2; lr=8e-5; start_epoch=0; num_queries=100

export EXP_DIR backbone epochs batch_size num_workers num_frames lr start_epoch num_queries

# CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 tools/run_dist_launch.sh 1 configs/swinb_train_single.sh 

CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 tools/run_dist_launch.sh 1 configs/swinb_eval_single.sh  