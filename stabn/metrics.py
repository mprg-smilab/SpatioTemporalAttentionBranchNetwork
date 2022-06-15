#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics import confusion_matrix


class ConfusionMatrix(object):
    """ConfusionMatrix
    
    Calculates a confusion matrix.

    NOTE:
        The shape of confusion matrix:
            y-axis (vertical): true
            x-axis (horizontal): predicted
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def clear(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, label_trues, label_preds):
        self.confusion_matrix += confusion_matrix(label_trues, label_preds, labels=list(range(self.num_classes)))

    def get_score(self):
        accuracy_top1 = self.confusion_matrix.diagonal().sum() / self.confusion_matrix.sum()
        class_score = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(dim=1)
        return {'top1_acc': accuracy_top1, 'cls_score': class_score}

    def get_conf_mat(self):
        return self.confusion_matrix
