#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np


class GroupColorJitter(object):
    def __init__(self, color_space_aug=False, alphastd=0.1, eigval=None, eigvec=None):
        if eigval is None:
            # note that the data range should be [0, 255] and channel order is BGR
            self.eigval = np.array([55.46, 4.794, 1.148])
        if eigvec is None:
            self.eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]])
        self.alphastd = alphastd
        self.color_space_aug = color_space_aug

    @staticmethod
    def brightnetss(img, delta):
        if random.uniform(0, 1) > 0.5:
            delta = np.array(delta).astype(np.float32)
            img = img + delta
        return img

    @staticmethod
    def contrast(img, alpha):
        if random.uniform(0, 1) > 0.5:
            alpha = np.array(alpha).astype(np.float32)
            img = img * alpha
        return img

    @staticmethod
    def saturation(img, alpha):
        if random.uniform(0, 1) > 0.5:
            gray = img * np.array([0.299, 0.587, 0.114]).astype(np.float32)
            gray = np.sum(gray, 2, keepdims=True)
            gray *= (1.0 - alpha)
            img = img * alpha
            img = img + gray
        return img

    @staticmethod
    def hue(img, alpha):
        if random.uniform(0, 1) > 0.5:
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            t = np.array(t).astype(np.float32)
            img = np.dot(img, t)
        return img

    def __call__(self, img_group):
        if self.color_space_aug:
            bright_delta = np.random.uniform(-32, 32)
            contrast_alpha = np.random.uniform(0.6, 1.4)
            saturation_alpha = np.random.uniform(0.6, 1.4)
            hue_alpha = random.uniform(-18, 18)
            out = []
            for img in img_group:
                img = self.brightnetss(img, delta=bright_delta)
                if random.uniform(0, 1) > 0.5:
                    img = self.contrast(img, alpha=contrast_alpha)
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                else:
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                    img = self.contrast(img, alpha=contrast_alpha)
                out.append(img)
            img_group = np.asarray(out)

        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.array(np.dot(self.eigvec * alpha, self.eigval)).astype(np.float32)
        bgr = np.expand_dims(np.expand_dims(rgb[::-1], 0), 0)
        output = [img + bgr for img in img_group]
        return np.asarray(output)
