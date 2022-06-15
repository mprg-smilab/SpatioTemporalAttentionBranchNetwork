#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""resnet.py

3D-ResNet.
This script is implemented on the basis of https://github.com/kenshohara/video-classification-3d-cnn-pytorch.
We are grateful for the author.
"""

import math
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def load_2d_pretrain_model(model, model_name, verbose=False):
    """"Copy parameters of 2D ResNet trained with ImageNet dataset."""

    state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
    keys = state_dict.keys()

    print("Copy model params. of 2D ResNet ...")
    for name, module in model.named_modules():
        ## params. of Conv2d
        if isinstance(module, nn.Conv3d) and name + ".weight" in keys:
            if verbose: print("  copy params:", name)
            new_weight = state_dict[name + ".weight"].unsqueeze(2).expand_as(module.weight) / module.weight.data.shape[2]
            module.weight.data.copy_(new_weight)
            if name + ".bias" in keys:
                if verbose: print("    copy bias:", name)
                new_bias = state_dict[name + ".bias"]
                module.bias.data.copy_(new_bias)
        ## params of BatchNorm2d
        elif isinstance(module, nn.BatchNorm3d) and \
            name+".running_mean" in keys and name+".running_var" in keys and \
            name+".weight" in keys and name+".bias" in keys:
                if verbose: print("  copy params:", name)
                module.weight.data.copy_(state_dict[name + ".weight"])
                module.bias.data.copy_(state_dict[name + ".bias"])
                module.running_mean.data.copy_(state_dict[name + ".running_mean"])
                module.running_var.data.copy_(state_dict[name + ".running_var"])
    
    return model


def resnet18(pretrain_2d=False, **kwargs):
    """Constructs a 3D ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet18')
    return model


def resnet34(pretrain_2d=False, **kwargs):
    """Constructs a 3D ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet34')
    return model


def resnet50(pretrain_2d=False, **kwargs):
    """Constructs a 3D ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet50')
    return model


def resnet101(pretrain_2d=False, **kwargs):
    """Constructs a 3D ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet101')
    return model


def resnet152(pretrain_2d=False, **kwargs):
    """Constructs a 3D ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet152')
    return model


if __name__ == '__main__':

    ### something-something setting
    n_class = 174
    image_size = 224
    frame_length = 32

    ### kinetics setting
    # n_class = 400
    # image_size = 112
    # frame_length = 16

    print("debug ...")
    print("    number of classes:", n_class)
    print("    image size:", image_size)
    print("    frame length:", frame_length)

    input = torch.zeros([2, 3, frame_length, image_size, image_size], dtype=torch.float32)

    model18 = resnet18(pretrain_2d=True, num_classes=n_class, sample_size=image_size, sample_duration=frame_length)
    output18 = model18(input)
    print(output18.size())

    model34 = resnet34(pretrain_2d=True, num_classes=n_class, sample_size=image_size, sample_duration=frame_length)
    output34 = model34(input)
    print(output34.size())

    model50 = resnet50(pretrain_2d=True, num_classes=n_class, sample_size=image_size, sample_duration=frame_length)
    output50 = model50(input)
    print(output50.size())

    model101 = resnet101(pretrain_2d=True, num_classes=n_class, sample_size=image_size, sample_duration=frame_length)
    output101 = model101(input)
    print(output101.size())

    model152 = resnet152(pretrain_2d=True, num_classes=n_class, sample_size=image_size, sample_duration=frame_length)
    output152 = model152(input)
    print(output152.size())
