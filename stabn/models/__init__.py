#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_abn_v1 import abn_resnet18, abn_resnet34, abn_resnet50, abn_resnet101, abn_resnet152
from .resnet_abn_v2 import abn_resnet18_v2, abn_resnet34_v2, abn_resnet50_v2, abn_resnet101_v2, abn_resnet152_v2


MODEL_NAMES = (
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'abn_resnet18', 'abn_resnet34', 'abn_resnet50', 'abn_resnet101', 'abn_resnet152',
    'abn_resnet18_v2', 'abn_resnet34_v2', 'abn_resnet50_v2', 'abn_resnet101_v2', 'abn_resnet152_v2'
)


def load_model(model_name, num_classes, sample_size, sample_duration, dout_ratio, pretrain_2d=True):
    assert model_name in MODEL_NAMES, "ERROR: model name %s does not exists." % model_name
    print("build network model: %s" % model_name)

    ### 3D-ResNet
    if model_name == 'resnet18':
        model = resnet18(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'resnet34':
        model = resnet34(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'resnet50':
        model = resnet50(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'resnet101':
        model = resnet101(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'resnet152':
        model = resnet152(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    ### ST-ABN (v1; Mitsuhara original)
    elif model_name == 'abn_resnet18':
        model = abn_resnet18(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'abn_resnet34':
        model = abn_resnet34(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'abn_resnet50':
        model = abn_resnet50(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'abn_resnet101':
        model = abn_resnet101(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'abn_resnet152':
        model = abn_resnet152(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    ### ST-ABN (v2)
    elif model_name == 'abn_resnet18_v2':
        model = abn_resnet18_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'abn_resnet34_v2':
        model = abn_resnet34_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'abn_resnet50_v2':
        model = abn_resnet50_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'abn_resnet101_v2':
        model = abn_resnet101_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)
    elif model_name == 'abn_resnet152_v2':
        model = abn_resnet152_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration, dropout_ratio=dout_ratio)

    return model
