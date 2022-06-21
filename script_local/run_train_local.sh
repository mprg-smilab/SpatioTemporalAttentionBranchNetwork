#!/bin/bash


# -----------------------------------------------
# Data Parallel
# -----------------------------------------------
### 3D ResNet 50
python3 train_dp.py --model resnet50 --pretrained \
    --logdir ./runs/res50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --use_nesterov --num_workers 188 \
    --gpu_id 0,1,2,3,4,5,6,7


### ST-ABN (ResNet50)
python3 train_dp.py --model abn_resnet50 --pretrained \
    --logdir ./runs/stabn_res50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --use_nesterov --num_workers 188 \
    --gpu_id 0,1,2,3,4,5,6,7


# -----------------------------------------------
# Distributed Data Parallel (DPP)
# -----------------------------------------------
### ST-ABN (ResNet50)
python3 -m torch.distributed.launch \
    --nproc_per_node=8 --master_addr="localhost" --master_port=1234 train_ddp.py \
    --model abn_resnet50 --pretrained \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --logdir ./runs/abn_resnet50 \
    --use_nesterov --num_workers 16


### Resuming ST-ABN (ResNet50)
python3 -m torch.distributed.launch \
    --nproc_per_node=8 --master_addr="localhost" --master_port=1234 train_ddp.py \
    --model abn_resnet50 --pretrained \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --logdir ./runs/abn_resnet50 \
    --use_nesterov --num_workers 16 \
    --resume checkpoint-latest.pt


# -----------------------------------------------
# MEMO
# -----------------------------------------------
### warning
# WARNING:torch.distributed.run:--use_env is deprecated and will be removed in future releases.
#  Please read local_rank from `os.environ('LOCAL_RANK')` instead.
# INFO:torch.distributed.launcher.api:Starting elastic_operator with launch configs:
