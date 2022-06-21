#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""attention.py

utility functions for visualizing attention maps
"""

import math
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# TODO: define a function
# def save_st_attention_minibatch(videos, sp_attentions, tp_attentions, save_names):


def save_st_attention(video, attention_sp, attention_tp, save_name):
    T, H, W, _ = video.shape
    _n_col = 8
    _n_row = math.ceil(T / _n_col)

    dst_att_map = []
    for v, a_sp, a_tp in zip(video, attention_sp, attention_tp.squeeze()):
        att_map_tmp = write_spatial_attention(v, a_sp)
        att_map_tmp = write_temporal_attention(att_map_tmp, a_tp)
        dst_att_map.append(att_map_tmp)

    fig, ax = plt.subplots(_n_row, _n_col, figsize=(10, 1.5 * _n_row))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(_n_row):
        for j in range(_n_col):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(dst_att_map[_n_col * i + j], cmap="bone")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def write_spatial_attention(frame, attention_sp):
    """make a single attention map image"""
    H, W, _ = frame.shape
    attention_sp *= 255
    att_map = cv2.applyColorMap(attention_sp.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    att_map = cv2.resize(att_map, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame.astype(np.uint8), 0.6, att_map, 0.4, 0)


def write_temporal_attention(frame, attention_tp, bar_width=20, offset=0):
    """make attention map images with temporal attention"""
    H, W, _ = frame.shape
    tp_map = np.ones([bar_width, W], dtype=np.float32) * (255.0 * attention_tp)
    tp_map = cv2.applyColorMap(tp_map.astype(np.uint8), cv2.COLORMAP_JET)
    if offset > 0:
        return cv2.vconcat((tp_map, np.ones([offset, W], dtype=np.uint8) * 255, frame))
    else:
        return cv2.vconcat((tp_map, frame))
