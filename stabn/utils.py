#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import torch


### variables for training
LOG_STEP = 200
CHECKPOINT_STEP = 10


### variables for evaluation (demo)
RESULT_DIR_TRAIN = "result_train"
RESULT_DIR_VAL = "result_val"
RESULT_DIR_TEST = "result_test"


def save_checkpoint(save_filename, model, optimizer, scheduler, best_score, epoch, iteration):
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'best_score': best_score,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_filename)


def load_checkpoint(load_filename, model, optimizer=None, scheduler=None):
    _ckpt = torch.load(load_filename)
    model.load_state_dict(_ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(_ckpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(_ckpt['scheduler_state_dict'])

    return model, optimizer, scheduler, _ckpt['epoch'], _ckpt['iteration'], _ckpt['best_score']


def save_args(save_filename, args):
    with open(save_filename, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def load_args(load_filename):
    with open(load_filename, 'r') as f:
        args = json.load(f)
    return args
