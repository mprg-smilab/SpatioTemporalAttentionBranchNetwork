#!/bin/bash


### 3D ResNet 50
python3 eval.py --logdir ./runs/res50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --gpu_id 0,1,2,3,4,5,6,7


### ST-ABN (ResNet50)
python3 eval.py --logdir ./runs/abn_res50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --save_attention

python3 eval.py --logdir ./runs/abn_res50 \
    --video_data_dir /raid/hirakawa/dataset/something-something-v2/frame \
    --train_label_file /raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt \
    --val_label_file /raid/hirakawa/dataset/something-something-v2/anno/val_videofolder.txt \
    --gpu_id 0,1,2,3,4,5,6,7 \
    --multi_segment
