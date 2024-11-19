# Drone-to-Drone-Detection

Steps to run the training script

0. Start the docker container:
    (a) nvidia-docker run --rm -it -v <path to DDoD folder>:/workspace <path to docker image> bash
    (eg. nvidia-docker run --rm -it -v /raid/ai20resch11003/Sairam/DDoD:/workspace pytorch/pytorch:ddod_1 bash)
    (b) Kill a docker container process:
        >> docker container ls (to find the docker id)
        >> docker container kill <ID>

1. Modify the Config file (swinb_train_single.sh) as per the requirements
    - EXP_DIR (stores the checkpoints, train and test logs)
2. datasets/vid_single.py : Has the code to prepare the data. 
    - Set dataset name: Fl or NPS and Set the Flags for temporal_data_loading
3. Commands: 

    CUDA_VISIBLE_DEVICES=x,y GPUS_PER_NODE=n tools/run_dist_launch.sh n configs/swinb_train_single.sh 

    CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 tools/run_dist_launch.sh 1 configs/swinb_train_single.sh 

    CUDA_VISIBLE_DEVICES=2,3 GPUS_PER_NODE=2 tools/run_dist_launch.sh 2 configs/swinb_train_single.sh 


Other important files:
1. dataset_dets.py: Set the Flags for temporal_data_loading and grad_accumulation
2. engine_single.py: Has code for train_one_epoch. 
. models/deformable_detr_single.py has code for the main model where we can set flags for all the model variations we want to try like 
    'temporal_data_fusion', 'visualizing_preds' and all
