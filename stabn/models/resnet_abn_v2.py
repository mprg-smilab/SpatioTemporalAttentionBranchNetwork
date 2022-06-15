#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""resnet_abn_v2.py

ST-ABN with 3D-ResNet backbone. V2 has a modified backbone architecture from v1.
The architecture of 3D-ResNet is modified for keeping the temporal dimention of feature map.

Difference between ST-ABN v1 and v2:
    The v1 architecture (original implementation by M. Mitsuhara) keeps the temporal feature
    dimension. Meanwhile, v2 architecture slightly shrink the temporal feature dimension.
"""

import copy
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, temporal_ksize=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=(1, stride, stride), padding=1, bias=False)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, temporal_ksize=1):
        super(Bottleneck, self).__init__()
        if temporal_ksize == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        else:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(temporal_ksize, 1, 1),
                                   padding=(int((temporal_ksize - 1) / 2), 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                               padding=(0, 1, 1), bias=False)
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


class ABNResNet(nn.Module):

    def __init__(self, block, layers, sample_duration, num_classes=400, verbose=False):
        self.inplanes = 64
        super(ABNResNet, self).__init__()

        # sample_size:     画像の縦横サイズ（のはず）
        # sample_duration: 動画フレーム数（のはず）
        self.verbose = verbose

        # feature extractor -----------
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                               padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temporal_ksize=3)

        # perception branch -----------
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temporal_ksize=3)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # attention branch ------------
        # common convolutional blocks
        self.inplanes = int(self.inplanes / 2)
        self.layer_att_common = self._make_layer(block, 512, layers[3], stride=1, temporal_ksize=3)

        self.conv_att_common = nn.Sequential(
            nn.BatchNorm3d(512 * block.expansion),
            nn.Conv3d(512 * block.expansion, num_classes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(num_classes),
            nn.ReLU(inplace=True)
        )

        # output of attention branch (classification score)
        self.conv_att_out = nn.Sequential(
            nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveAvgPool3d((sample_duration, 1, 1))
        )
        self.fc_att_out = nn.Linear(num_classes * sample_duration, 174)

        # spatial attention branch
        self.conv_spatial_att = nn.Sequential(
            nn.Conv3d(num_classes, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        # temporal attention branch
        self.conv_temp_att1 = nn.Sequential(
            nn.Conv3d(num_classes, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        # (intermediate process: get tensor shape and reshape tensor)
        self.conv_temp_att2 = nn.Sequential(
            nn.Conv2d(sample_duration, sample_duration, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        # (intermediate process: reshape tensor)
        self.conv_temp_att3 = nn.Sequential(
            nn.Linear(sample_duration, sample_duration),
            nn.ReLU(inplace=True),
            nn.Linear(sample_duration, sample_duration),
            nn.Sigmoid()
        )
        # (intermediate process: reshape tensor and forward attention mechanism)

        # attention mechanism ---------
        self.conv_att_mechanism = nn.Sequential(
            nn.Conv3d(512 * block.expansion, int(512 * block.expansion / 2), kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(int(512 * block.expansion / 2))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, temporal_ksize=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temporal_ksize=temporal_ksize))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, temporal_ksize=temporal_ksize))

        return nn.Sequential(*layers)

    def forward(self, x):

        # feature extractor -----------
        if self.verbose: print("feature extractor")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # attention branch ------------
        # common convolutional blocks
        if self.verbose: print("attention branch")
        x_att = self.layer_att_common(x)
        x_att = self.conv_att_common(x_att)

        # output of attention branch (classification score)
        if self.verbose: print("    output of att branch")
        out_att = self.conv_att_out(x_att)
        out_att = out_att.view(out_att.size(0), -1)
        out_att = self.fc_att_out(out_att)
        
        # spatial attention branch
        if self.verbose: print("    spatial att branch")
        spatial_att = self.conv_spatial_att(x_att)

        # temporal attention branch
        if self.verbose: print("    temporal att branch")
        x_att_temp = self.conv_temp_att1(x_att)
        x_att_temp = x_att_temp.view(x_att_temp.size(0), x_att_temp.size(2), x_att_temp.size(3), x_att_temp.size(4))
        x_att_temp = self.conv_temp_att2(x_att_temp)
        x_att_temp = x_att_temp.view(x_att_temp.size(0), -1)
        x_att_temp = self.conv_temp_att3(x_att_temp)
        temporal_att = x_att_temp.view(x_att_temp.size(0), 1, x_att_temp.size(1), 1, 1)

        # attention mechanism ---------
        if self.verbose: print("attention mechanism")
        # spatial
        x_sp_att_mechanism = x * torch.add(spatial_att, 1)
        # temporal
        x_tp_att_mechanism = x * temporal_att
        # concat and convolution
        x_att_mechanism = torch.cat((x_sp_att_mechanism, x_tp_att_mechanism), dim=1)
        x_att_mechanism = self.conv_att_mechanism(x_att_mechanism)

        # perception branch -----------
        if self.verbose: print("perception branch")
        x_per = self.layer4(x_att_mechanism)
        x_per = self.avgpool(x_per)
        x_per = x_per.view(x_per.size(0), -1)
        out_per = self.fc(x_per)

        return out_per, out_att, spatial_att, temporal_att


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


def abn_resnet18_v2(pretrain_2d=False, **kwargs):
    """Constructs an ST Attention Branch Network with 3D ResNet-18 model (v2)."""
    model = ABNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet18')
    return model


def abn_resnet34_v2(pretrain_2d=False, **kwargs):
    """Constructs an ST Attention Branch Network with 3D ResNet-34 model (v2)."""
    model = ABNResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet34')
    return model


def abn_resnet50_v2(pretrain_2d=False, **kwargs):
    """Constructs an ST Attention Branch Network with 3D ResNet-50 model (v2)."""
    model = ABNResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet50')
    return model


def abn_resnet101_v2(pretrain_2d=False, **kwargs):
    """Constructs an ST Attention Branch Network with 3D ResNet-101 model (v2)."""
    model = ABNResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    model = load_2d_pretrain_model(model, 'resnet101')
    return model


def abn_resnet152_v2(pretrain_2d=False, **kwargs):
    """Constructs an ST Attention Branch Network with 3D ResNet-101 model (v2)."""
    model = ABNResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
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

    model18 = abn_resnet18_v2(pretrain_2d=True, num_classes=n_class, sample_duration=frame_length)
    output18 = model18(input)
    print(output18[0].size(), output18[1].size(), output18[2].size(), output18[3].size())

    model34 = abn_resnet34_v2(pretrain_2d=True, num_classes=n_class, sample_duration=frame_length)
    output34 = model34(input)
    print(output34[0].size(), output34[1].size(), output34[2].size(), output34[3].size())

    model50 = abn_resnet50_v2(pretrain_2d=True, num_classes=n_class, sample_duration=frame_length)
    output50 = model50(input)
    print(output50[0].size(), output50[1].size(), output50[2].size(), output50[3].size())

    model101 = abn_resnet101_v2(pretrain_2d=True, num_classes=n_class, sample_duration=frame_length)
    output101 = model101(input)
    print(output101[0].size(), output101[1].size(), output101[2].size(), output101[3].size())

    model152 = abn_resnet152_v2(pretrain_2d=True, num_classes=n_class, sample_duration=frame_length)
    output152 = model152(input)
    print(output152[0].size(), output152[1].size(), output152[2].size(), output152[3].size())
