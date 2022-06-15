#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
eval.py

This script evaluates a trained ST-ABN or 3D ResNet with Something-Something V2 Dataset.
"""


import os
import random
import json
from time import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from stabn.models import MODEL_NAMES, load_model
from stabn.datasets import SomethingSomethingV2
from stabn.utils import load_checkpoint, load_args, RESULT_DIR_TRAIN, RESULT_DIR_VAL, RESULT_DIR_TEST
from stabn.attention import save_st_attention


def parser():
    arg_parser = ArgumentParser(add_help=True)

    ### basic settings
    arg_parser.add_argument('--logdir', type=str, required=True, help='directory stored trained model and settings')
    arg_parser.add_argument('--arg_file', type=str, default='args.json', help='json file name saved training settings')
    arg_parser.add_argument('--resume', type=str, default='checkpoint-best.pt', help='trained model file')

    ### evaluation settings
    arg_parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size for evaluation')
    arg_parser.add_argument('--multi_segment', action='store_true', help='use multiple frame segments for inference')
    arg_parser.add_argument('--num_workers', type=int, default=32, help='the number of multiprocess workers for data loader')
    arg_parser.add_argument('--save_attention', action='store_true', help='save attention maps')

    ### dataset path
    arg_parser.add_argument('--video_data_dir', type=str, required=True, help='path to video data directory')
    arg_parser.add_argument('--train_label_file', type=str, required=True, help='path to train label text file')
    arg_parser.add_argument('--val_label_file', type=str, required=True, help='path to validation label text file')

    ### GPU settings
    arg_parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    return arg_parser.parse_args()


def main():
    args = parser()

    # GPU (device) settings ###############################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    print("use cuda:", use_cuda)
    assert use_cuda, "Please use GPUs for evaluation."

    # load train args #####################################
    train_args = load_args(os.path.join(args.logdir, args.arg_file))

    # dataset #############################################
    print("load dataset")

    # check if use multiple segments for inference
    num_segment = 2 if args.multi_segment else 1

    train_dataset = SomethingSomethingV2(
        video_data_dir=args.video_data_dir, label_file=args.train_label_file, train_mode=False,
        frame_size=train_args['frame_size'], frame_length=train_args['frame_length'],
        num_segment=num_segment
    )
    val_dataset = SomethingSomethingV2(
        video_data_dir=args.video_data_dir, label_file=args.val_label_file, train_mode=False,
        frame_size=train_args['frame_size'], frame_length=train_args['frame_length'],
        num_segment=num_segment
    )

    kwargs = {'num_workers': args.num_workers, 'pin_memory': False}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # network model #######################################
    _is_abn = True if "abn_" in train_args['model'] else False
    model = load_model(
         model_name=train_args['model'], num_classes=train_dataset.NUM_CLASSES,
        sample_size=train_args['frame_size'], sample_duration=train_args['frame_length'], pretrain_2d=False
    )

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    
    # resume ##############################################
    assert os.path.exists(os.path.join(args.logdir, args.resume))
    print("    load checkpont:", os.path.join(args.logdir, args.resume))
    model = load_checkpoint(os.path.join(args.logdir, args.resume), model)[0]

    #######################################################
    # the beginning of evaluation
    #######################################################

    ### train data
    if args.train_label_file is not None:
        print("evaluate train data ...")
        evaluation(model, train_loader, RESULT_DIR_TRAIN, args, use_cuda, _is_abn)

    ### val data
    if args.val_label_file is not None:
        print("evaluate validation data ...")
        evaluation(model, val_loader, RESULT_DIR_VAL, args, use_cuda, _is_abn)

    #######################################################
    # the end of evaluation
    #######################################################

    print("evaluation; done.")


def evaluation(model, data_loader, result_dir_name, args, use_cuda, is_abn):

    ### make result directory in logdir
    result_dir = os.path.join(args.logdir, result_dir_name)
    os.makedirs(result_dir, exist_ok=True)

    count_per_top1, count_att_top1 = 0, 0
    count_per_top5, count_att_top5 = 0, 0

    ### evaluation
    model.eval()
    with torch.no_grad():
        for video, label, video_name in data_loader:
            video, label = video.cuda(), label.cuda()
            output = model(video)

            # count for top-1 accuracy
            if is_abn:
                count_per_top1 += torch.sum(output[0].argmax(dim=1) == label)
                count_att_top1 += torch.sum(output[1].argmax(dim=1) == label)
            else:
                count_per_top1 += torch.sum(output.argmax(dim=1) == label)

            # count for top-5 accuracy
            if is_abn:
                sorted_output_per = output[0].argsort(dim=1, descending=True)
                sorted_output_att = output[1].argsort(dim=1, descending=True)
                for o_per, o_att, l in zip(sorted_output_per, sorted_output_att, label):
                    if l in o_per[0:5]:
                        count_per_top5 += 1
                    if l in o_att[0:5]:
                        count_att_top5 += 1
            else:
                sorted_output_per = output.argsort(dim=1, descending=True)
                for o_per, l in zip(sorted_output_per, label):
                    if l in o_per[0:5]:
                        count_per_top5 += 1

            # save attention maps (as image)
            if is_abn and not args.multi_segment and args.save_attention:
                video_array = video.data.cpu().numpy()
                attention_spatial  = output[2].data.cpu().numpy().squeeze()  # (B, T, H, W)
                attention_temporal = output[3].data.cpu().numpy().squeeze()  # (B, T)
                for v, att_sp, att_tp, v_id in zip(video_array, attention_spatial, attention_temporal, video_name):
                    save_st_attention(data_loader.dataset._denormalize(v), att_sp, att_tp, os.path.join(result_dir, v_id + ".png"))

    ### top-1 and top-5 accuracy
    acc_per_top1 = count_per_top1 / len(data_loader.dataset)
    acc_att_top1 = count_att_top1 / len(data_loader.dataset)
    acc_per_top5 = count_per_top5 / len(data_loader.dataset)
    acc_att_top5 = count_att_top5 / len(data_loader.dataset)

    # print
    print("  top-1 accuracy:")
    print("    accuracy (per):", acc_per_top1)
    print("    accuracy (att):", acc_att_top1)
    print("  top-5 accuracy:")
    print("    accuracy (per):", acc_per_top5)
    print("    accuracy (att):", acc_att_top5)

    # write for text file
    accuracy_dict = {
        'top-1_accuracy': {'per': acc_per_top1, 'att': acc_att_top1},
        'top-5_accuracy': {'per': acc_per_top5, 'att': acc_att_top5}
    }
    with open(os.path.join(result_dir, "accuracy.json"), 'w') as f:
        json.dump(accuracy_dict, f, indent=4)


if __name__ == '__main__':
    main()
