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


def load_model(model_name, num_classes, sample_size, sample_duration, pretrain_2d=True):
    assert model_name in MODEL_NAMES, "ERROR: model name %s does not exists." % model_name
    print("build network model: %s" % model_name)

    ### 3D-ResNet
    if model_name == 'resnet18':
        model = resnet18(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    elif model_name == 'resnet34':
        model = resnet34(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    elif model_name == 'resnet50':
        model = resnet50(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    elif model_name == 'resnet101':
        model = resnet101(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    elif model_name == 'resnet152':
        model = resnet152(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    ### ST-ABN (v1; Mitsuhara original)
    elif model_name == 'abn_resnet18':
        model = abn_resnet18(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet34':
        model = abn_resnet34(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet50':
        model = abn_resnet50(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet101':
        model = abn_resnet101(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet152':
        model = abn_resnet152(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    ### ST-ABN (v2)
    elif model_name == 'abn_resnet18_v2':
        model = abn_resnet18_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet34_v2':
        model = abn_resnet34_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet50_v2':
        model = abn_resnet50_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet101_v2':
        model = abn_resnet101_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet152_v2':
        model = abn_resnet152_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)

    return model


def load_resnet(model_name, num_classes, sample_size, sample_duration, pretrain_2d=True):
    RESNET_MODEL_NAMES = ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
    assert model_name in RESNET_MODEL_NAMES, "ERROR: model name %s does not exists." % model_name
    print("build 2d-ResNet model: %s" % model_name)

    if model_name == 'resnet18':
        model = resnet18(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    elif model_name == 'resnet34':
        model = resnet34(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    elif model_name == 'resnet50':
        model = resnet50(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    elif model_name == 'resnet101':
        model = resnet101(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    elif model_name == 'resnet152':
        model = resnet152(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_size=sample_size, sample_duration=sample_duration)
    
    return model


def load_stabn(model_name, num_classes, sample_duration, pretrain_2d=True):
    STABN_MODEL_NAMES = ('abn_resnet18', 'abn_resnet34', 'abn_resnet50', 'abn_resnet101', 'abn_resnet152')
    assert model_name in STABN_MODEL_NAMES, "ERROR: model name %s does not exists." % model_name
    print("build ST-ABN model: %s" % model_name)

    if model_name == 'abn_resnet18':
        model = abn_resnet18(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet34':
        model = abn_resnet34(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet50':
        model = abn_resnet50(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet101':
        model = abn_resnet101(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet152':
        model = abn_resnet152(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)


def load_stabn_v2(model_name, num_classes, sample_duration, pretrain_2d=True):
    STABN_V2_MODEL_NAMES = ('abn_resnet18_2', 'abn_resnet34_2', 'abn_resnet50_2', 'abn_resnet101_2', 'abn_resnet152_2')
    assert model_name in STABN_V2_MODEL_NAMES, "ERROR: model name %s does not exists." % model_name
    print("build ST-ABN model: %s" % model_name)

    if model_name == 'abn_resnet18_v2':
        model = abn_resnet18_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet34_v2':
        model = abn_resnet34_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet50_v2':
        model = abn_resnet50_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet101_v2':
        model = abn_resnet101_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)
    elif model_name == 'abn_resnet152_v2':
        model = abn_resnet152_v2(pretrain_2d=pretrain_2d, num_classes=num_classes, sample_duration=sample_duration)

    return model
