#!/bin/bash


# -----------------------------------------------
# Data Parallel (DP; train_dp.py)
# NOTE: We reccomend to use DDP training program described below
# -----------------------------------------------


### 3D ResNet 50
python3 train_dp.py --model resnet50 --pretrained --logdir ./runs/res50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --use_nesterov --num_workers 188 --gpu_id 0,1,2,3,4,5,6,7


### ST-ABN (ResNet50)
python3 train_dp.py --model abn_resnet50 --pretrained --logdir ./runs/stabn_res50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --use_nesterov --num_workers 188 --gpu_id 0,1,2,3,4,5,6,7
