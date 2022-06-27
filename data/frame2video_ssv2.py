#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
frame2video_ssv2.py

This script converts video frames of JPEG image to mp4 video file.
"""


import os
from glob import glob
import cv2
import multiprocessing as mp


### path to directories
FRAME_DIR = "/raid/hirakawa/dataset/something-something-v2/frame"
VIDEO_DIR = "/raid/hirakawa/dataset/something-something-v2/video"
### number of threads for processing
NUM_THREADS = 128


def make_mp4(frame_directory, save_video_name):

    frame_list = glob(os.path.join(frame_directory, "*.jpg"))
    frame_list.sort()

    _tmp_frame = cv2.imread(frame_list[0])
    H, W, C = _tmp_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(save_video_name, fourcc, 16, (W, H))

    for fl in frame_list:
        frame = cv2.imread(fl)
        video.write(frame)

    video.release()


def child_process(frame_dir_list):
    for fd in frame_dir_list:
        outname = os.path.join(VIDEO_DIR, fd.split('/')[-1] + ".mp4")
        make_mp4(fd, outname)


def split_n_lists(input_list, n):
    size = len(input_list)

    dst_list = []
    for i in range(n):
        start = int((size*i/n))
        end = int((size*(i+1)/n))
        dst_list.append(input_list[start:end])

    return dst_list


def main():

    os.makedirs(VIDEO_DIR, exist_ok=True)

    dir_names = glob(os.path.join(FRAME_DIR, "*"))
    dir_names.sort()

    split_dir_names = split_n_lists(dir_names, NUM_THREADS)

    process_list = []

    for sdn in split_dir_names:
        process_list.append(mp.Process(target=child_process, args=(sdn, )))

    print("The number of multiprocess:", len(process_list))
    
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()
    
    print("done.")


if __name__ == '__main__':
    main()
