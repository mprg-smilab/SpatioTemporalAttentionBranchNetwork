#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
train_dp.py

This script train an ST-ABN or 3D ResNet with Something-Something V2 Dataset.
For using multiple GPUs, this script uses Data Parallel (DP) of PyTorch.
"""


import os
import json
from time import time
from argparse import ArgumentParser

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from stabn.models import MODEL_NAMES, load_model
from stabn.datasets import SomethingSomethingV2
from stabn.utils import load_checkpoint, save_checkpoint, save_args, load_args, LOG_STEP, CHECKPOINT_STEP


def parser():
    arg_parser = ArgumentParser(add_help=True)

    ### network settings
    arg_parser.add_argument('--model', type=str, default='abn_resnet50', choices=MODEL_NAMES, help='network model')
    arg_parser.add_argument('--pretrained', action='store_true', help='use pretrained network model as initial parameter')

    ### dataset path
    arg_parser.add_argument('--video_data_dir', type=str, required=True, help='path to video data directory')
    arg_parser.add_argument('--train_label_file', type=str, required=True, help='path to train label text file')
    arg_parser.add_argument('--val_label_file', type=str, required=True, help='path to validation label text file')

    ### dataset settings
    arg_parser.add_argument('--frame_size', type=int, default=224, help='frame size of video clips')
    arg_parser.add_argument('--frame_length', type=int, default=32, help='frame length of a single video clips')
    arg_parser.add_argument('--dout_ratio', type=float, default=0.5, help='dropout ratio')

    ### traininig settings
    arg_parser.add_argument('--logdir', type=str, required=True, help='directory for storing train log and checkpoints')
    arg_parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    arg_parser.add_argument('--epochs', type=int, default=150, help='the number of training epochs')
    arg_parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    arg_parser.add_argument('--lr_steps', type=int, nargs='+', default=[75, 125], help='epochs to decrease learning rate')
    arg_parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD optimizer')
    arg_parser.add_argument('--wd', type=float, default=0.0005, help='weight decay of SGD optimizer')
    arg_parser.add_argument('--use_nesterov', action='store_true', help='use nesterov accelerated SGD')
    arg_parser.add_argument('--num_workers', type=int, default=12, help='the number of multiprocess workers for data loader')

    ### resume settings
    arg_parser.add_argument('--resume', type=str, default=None, help='filename of checkpoint for resuming the training')

    ### GPU settings
    arg_parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    args = arg_parser.parse_args()

    ### resume train args
    if args.resume is not None:
        _resume_path = os.path.join(args.logdir, args.resume)
        assert os.path.exists(_resume_path), "ERROR: resume file does not exist: %s" % _resume_path

        print("resume: duplicate arguments ...")
        _resume_args = load_args(os.path.join(args.logdir, "args.json"))
        args.model        = _resume_args['model']
        args.pretrained   = _resume_args['pretrained']
        args.frame_size   = _resume_args['frame_size']
        args.frame_length = _resume_args['frame_length']
        args.dout_ratio   = _resume_args['dout_ratio']
        args.batch_size   = _resume_args['batch_size']
        args.epochs       = _resume_args['epochs']
        args.lr           = _resume_args['lr']
        args.lr_steps     = _resume_args['lr_steps']
        args.momentum     = _resume_args['momentum']
        args.wd           = _resume_args['wd']
        args.use_nesterov = _resume_args['use_nesterov']

    return args


def main():
    args = parser()

    # GPU (device) settings ###############################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    print("use cuda:", use_cuda)

    # dataset #############################################
    print("load dataset")
    train_dataset = SomethingSomethingV2(
        video_data_dir=args.video_data_dir, label_file=args.train_label_file,
        train_mode=True, frame_size=args.frame_size, frame_length=args.frame_length, num_segment=1
    )
    val_dataset = SomethingSomethingV2(
        video_data_dir=args.video_data_dir, label_file=args.val_label_file,
        train_mode=False, frame_size=args.frame_size, frame_length=args.frame_length, num_segment=1
    )

    kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, **kwargs)

    # network model & loss ################################
    _is_abn = True if "abn_" in args.model else False
    model = load_model(
        model_name=args.model, num_classes=train_dataset.NUM_CLASSES,
        sample_size=args.frame_size, sample_duration=args.frame_length, dout_ratio=args.dout_ratio, pretrain_2d=args.pretrained
    )
    criterion = nn.CrossEntropyLoss()

    # optimizer ###########################################
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.use_nesterov)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)

    # CPU or GPU
    if use_cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    initial_epoch = 1
    iteration = 0
    best_score = 0.0
    loss_sum, loss_sum_att, loss_sum_per = 0.0, 0.0, 0.0

    # resume ##############################################
    if args.resume is not None:
        print("Load checkpoint for resuming a training ...")
        print("    checkpoint:", os.path.join(args.logdir, args.resume))
        model, optimizer, scheduler, initial_epoch, iteration, best_score = load_checkpoint(os.path.join(args.logdir, args.resume), model, optimizer, scheduler)
        initial_epoch += 1

    # tensorboardX ########################################
    writer = SummaryWriter(log_dir=args.logdir)
    log_dir = writer.file_writer.get_logdir()
    save_args(os.path.join(log_dir, 'args.json'), args)

    #######################################################
    # the beginning of train loop
    #######################################################
    _start = time()
    for epoch in range(initial_epoch, args.epochs + 1):
        print("epoch:", epoch)

        # train #######################
        model.train()
        for video, label, _ in train_loader:
            iteration += 1

            if use_cuda:
                video, label = video.cuda(), label.cuda()
            
            output = model(video)
            if _is_abn:
                loss_per = criterion(output[0], label)
                loss_att = criterion(output[1], label)
                loss = loss_per + loss_att
            else:
                loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if _is_abn:
                loss_sum_per += loss_per.item()
                loss_sum_att += loss_att.item()

            if iteration % LOG_STEP == 0:
                print("iteration: %06d loss: %0.8f (per: %0.8f, att: %0.8f) elapsed time: %0.1f" % (
                    iteration, loss_sum / LOG_STEP, loss_sum_per / LOG_STEP, loss_sum_att / LOG_STEP, time() - _start
                ))
                writer.add_scalar("loss/all", loss_sum / LOG_STEP, iteration)
                if _is_abn:
                    writer.add_scalar("loss/per", loss_sum_per / LOG_STEP, iteration)
                    writer.add_scalar("loss/att", loss_sum_att / LOG_STEP, iteration)
                loss_sum, loss_sum_att, loss_sum_per = 0.0, 0.0, 0.0

        # validation #######################
        print("evaluation ...")
        model.eval()
        count_per, count_att = 0, 0
        with torch.no_grad():
            for i, (video, label, _) in enumerate(val_loader):
                if use_cuda:
                    video, label = video.cuda(), label.cuda()

                output = model(video)
                if _is_abn:
                    count_per += torch.sum(output[0].argmax(dim=1) == label)
                    count_att += torch.sum(output[1].argmax(dim=1) == label)
                else:
                    count_per += torch.sum(output.argmax(dim=1) == label)

            acc_per = count_per / len(val_dataset)
            if _is_abn:
                acc_att = count_att / len(val_dataset)

        print("    accuracy (per):", acc_per.item())
        writer.add_scalar("accuracy/per", acc_per.item(), epoch)
        if _is_abn:
            print("    accuracy (att):", acc_att.item())
            writer.add_scalar("accuracy/att", acc_att.item(), epoch)

        # change learning rate ########
        scheduler.step()

        # save model ##################
        print("save model ...")
        ### 1. best validation accuracy
        if best_score < acc_per.item():
            print("    best score")
            best_score = acc_per.item()
            save_checkpoint(os.path.join(log_dir, "checkpoint-best.pt"), model, optimizer, scheduler, best_score, epoch, iteration)

        ### 2. at regular intervals
        if epoch % CHECKPOINT_STEP == 0:
            print("    regular intervals")
            save_checkpoint(os.path.join(log_dir, "checkpoint-%04d.pt" % epoch), model, optimizer, scheduler, best_score, epoch, iteration)

        ### 3. for resume (latest checkpoint)
        print("    latest")
        save_checkpoint(os.path.join(log_dir, "checkpoint-latest.pt"), model, optimizer, scheduler, best_score, epoch, iteration)

        print("epoch:", epoch, "; done.\n")
    #######################################################
    # the end of train loop
    #######################################################

    # save final model & close tensorboard writer
    print("save final model")
    save_checkpoint(os.path.join(log_dir, "checkpoint-final.pt"), model, optimizer, scheduler, best_score, epoch, iteration)
    writer.close()
    print("training; done.")


if __name__ == '__main__':
    main()
