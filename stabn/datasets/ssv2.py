#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ctypes.wintypes import PSIZE
from os.path import exists, join
import json
import numpy as np
import cv2
from torch.utils.data import Dataset

from .groupcolorjitter import GroupColorJitter


class SomethingSomethingV2(Dataset):
    """SomethingSomethingV2 Dataset Class
    
    Note:
        We assume that the channel order of loaded video frames is [B, G, R] based on
        the original implementation of ST-ABN and TPN repository. Especially, we use the
        color jitter implementation of the original TPN (mmaction), which assumes the
        input frame is BGR channel order.
    """

    def __init__(self, video_data_dir, label_file, train_mode=True, frame_size=224, frame_length=32, num_segment=1, verbose=False):
        super().__init__()

        ### FIXED PARMAMETERS
        self.NUM_CLASSES = 174
        self.FRAME_SKIP = 1

        ### mean and std with RGB order
        # self.MEAN = [123.675, 116.28, 103.53]
        # self.STD = [58.395, 57.12, 57.375]
        ### mean and std with BGR order
        self.MEAN = [103.53, 116.28, 123.675]
        self.STD = [57.375, 57.12, 58.395]

        # check file existance and assertion
        assert exists(video_data_dir)
        assert exists(label_file)

        ### set class variables
        self.video_data_dir = video_data_dir
        self.label_file = label_file
        self.train_mode = train_mode
        self.frame_size = frame_size
        self.frame_length = frame_length
        self.num_segment = num_segment

        ### augmentation class insntance
        self.colorjitter = GroupColorJitter(color_space_aug=True)

        ### load label file
        with open(self.label_file, 'r') as f:
            lines = list(map(lambda s: s.strip(), f.readlines()))
        self.labels = np.zeros((len(lines), 3), dtype=np.int32)
        for i, l in enumerate(lines):
            _vid_id, _f_len, _cls_id = l.split(' ')
            self.labels[i, 0] = int(_vid_id)
            self.labels[i, 1] = int(_f_len)
            self.labels[i, 2] = int(_cls_id)

        ### print details
        if verbose:
            print("Load Something-Something V2 Dataset:")
            print("    train mode:", self.train_mode)
            print("    the number of samples:", int(self.labels.shape[0]))

    def _get_frames(self, video_id, max_length):

        ### sample start frame index
        if max_length < self.frame_length:
            start_indices = np.zeros((self.num_segment,))
        else:
            if self.train_mode:
                start_indices = np.sort(np.random.randint(max_length - self.frame_length + 1,
                    size=self.num_segment))
            else:
                tick = (max_length - self.frame_length + 1) / float(self.num_segment)
                start_indices = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)])

        start_indices = start_indices + 1   # frame starts with 1 not 0
        start_indices = start_indices.astype(np.int32).tolist()

        frame_list = []
        for s_ind in start_indices:
            frames_tmp = []
            for i in range(0, self.frame_length):
                if s_ind + i <= max_length:
                    _f_tmp = np.array(cv2.imread(join(self.video_data_dir, str(video_id), "%06d.jpg" % (s_ind + i))))
                else:
                    _f_tmp = np.array(cv2.imread(join(self.video_data_dir, str(video_id), "%06d.jpg" % (s_ind))))
                frames_tmp.append(_f_tmp)
            frame_list.append(np.asarray(frames_tmp, dtype=np.float32))

        return frame_list

    def _crop_frames(self, frame):
        # NOTE: expected input data shape: [T, H, W, C] ndarray

        _, _height, _width, _ = frame.shape

        # random crop
        if self.train_mode:
            _y = np.random.randint(_height - self.frame_size + 1, size=1)[0]
            _x = np.random.randint(_width - self.frame_size + 1, size=1)[0]
            return frame[:, _y:_y+self.frame_size, _x:_x+self.frame_size, :]
        # center crop
        else:
            _y = int((_height / 2) - (self.frame_size / 2))
            _x = int((_width / 2) - (self.frame_size / 2))
            return frame[:, _y:_y+self.frame_size, _x:_x+self.frame_size, :]

    def _augmentation(self, frame):
        # expected input data shape: [T, H, W, C] ndarray

        # horizontal flip
        if np.random.rand(1)[0] > 0.5:
            frame = np.flip(frame, axis=1)

        # Color Jitter
        frame = self.colorjitter(frame)

        return frame

    def _normalize(self, frame):
        # NOTE: expected input data shape: [T, H, W, C] ndarray

        frame[:, :, :, 0] = (frame[:, :, :, 0] - self.MEAN[0]) / self.STD[0]
        frame[:, :, :, 1] = (frame[:, :, :, 1] - self.MEAN[1]) / self.STD[1]
        frame[:, :, :, 2] = (frame[:, :, :, 2] - self.MEAN[2]) / self.STD[2]
        return frame

    def _denormalize(self, frame, ch_order='CTHW'):
        # NOTE: expected input data shape: [T, H, W, C] ndarray

        if ch_order == 'CTHW':
            frame = frame.transpose(1, 2, 3, 0)

        frame[:, :, :, 0] = (frame[:, :, :, 0] * self.STD[0]) + self.MEAN[0]
        frame[:, :, :, 1] = (frame[:, :, :, 1] * self.STD[1]) + self.MEAN[1]
        frame[:, :, :, 2] = (frame[:, :, :, 2] * self.STD[2]) + self.MEAN[2]
        return frame

    def __getitem__(self, item):
        _vid_id = self.labels[item, 0]
        _f_len = self.labels[item, 1]
        _cls_id = self.labels[item, 2]

        # load video frames (THWC x num_segment)
        _video_frames = self._get_frames(_vid_id, _f_len)

        # preprocessing for video frames
        _dst_frames = []
        for _vf in _video_frames:
            # frame crop (THWC)
            _vf = self._crop_frames(_vf)
            # data augmetnation (THWC)
            if self.train_mode:
                _vf = self._augmentation(_vf)
            # normalize (THWC)
            _vf = self._normalize(_vf)
            # transpose frames (THWC --> CTHW)
            _vf = _vf.transpose(3, 0, 1, 2)

            _dst_frames.append(_vf.astype(np.float32).copy())

        return *_dst_frames, int(_cls_id), str(_vid_id)

    def __len__(self):
        return int(self.labels.shape[0])


if __name__ == '__main__':

    print("debug mode ...")

    import os
    import torch
    from torch.utils.data import DataLoader

    # GPU (device) settings ###############################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

    debug_dataset = SomethingSomethingV2(
        video_data_dir="/raid/hirakawa/dataset/something-something-v2/frame",
        label_file="/raid/hirakawa/dataset/something-something-v2/anno/train_videofolder.txt",
        train_mode=True, frame_size=224, frame_length=32, num_segment=2
    )

    for i in range(len(debug_dataset)):
        data = debug_dataset[i]
        print(data[0].shape, data[1].shape, data[2], data[3])
        
        if i == 2:
            break

    debug_loader = DataLoader(debug_dataset, batch_size=4, shuffle=True, **kwargs)

    for img1, img2, lab, vid_id in debug_loader:
        print(img1.size(), img2.size(), lab.size(), len(vid_id))
        exit(-1)
