#!/bin/bash


# -----------------------------------------------
# Distributed Data Parallel (DPP; train_ddp.py)
# -----------------------------------------------


### 3D ResNet 50
python3 -m torch.distributed.launch \
    --nproc_per_node=8 --master_addr="localhost" --master_port=1234 train_ddp.py \
    --model resnet50 --pretrained --logdir ./runs/res50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --use_nesterov --num_workers 16


### 3D ResNet 101
python3 -m torch.distributed.launch \
    --nproc_per_node=8 --master_addr="localhost" --master_port=1234 train_ddp.py \
    --model resnet101 --pretrained --logdir ./runs/res101 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --use_nesterov --num_workers 16


### ST-ABN (ResNet50)
python3 -m torch.distributed.launch \
    --nproc_per_node=8 --master_addr="localhost" --master_port=1234 train_ddp.py \
    --model abn_resnet50 --pretrained --logdir ./runs/abn_resnet50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --use_nesterov --num_workers 16


### Resuming a training of ST-ABN (ResNet50)
python3 -m torch.distributed.launch \
    --nproc_per_node=8 --master_addr="localhost" --master_port=1234 train_ddp.py \
    --model abn_resnet50 --pretrained --logdir ./runs/abn_resnet50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --use_nesterov --num_workers 16 \
    --resume checkpoint-latest.pt
