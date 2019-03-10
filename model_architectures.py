from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import tqdm
import os

import sys
import math

from collections import OrderedDict
import numpy as np


class SaliencyModel(nn.Module):

    def __init__(self):

        super(SaliencyModel, self).__init__()

        num_features = [32, 64, 64, 128, 256]
        num_dec_features = 32

        self.encoder1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_features[0], kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_features[0])),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(num_features[0], num_features[1], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(num_features[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2))
        ]))

        self.encoder2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(num_features[1], num_features[2], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(num_features[2])),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2))
        ]))

        self.encoder3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(num_features[2], num_features[3], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(num_features[3])),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=2)),
            ('conv4', nn.Conv2d(num_features[3], num_features[4], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(num_features[4])),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2))
        ]))

        self.decoder1 = nn.Sequential(OrderedDict([
            ('up1', nn.Upsample(scale_factor=2)),
            ('conv5', nn.Conv2d(num_features[1], num_dec_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm5', nn.BatchNorm2d(num_dec_features)),
            ('relu5', nn.ReLU(inplace=True)),
            ('final-conv1', nn.Conv2d(num_dec_features, 1, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

        self.decoder2 = nn.Sequential(OrderedDict([
            ('up2', nn.Upsample(scale_factor=2)),
            ('conv6', nn.Conv2d(num_features[2], num_dec_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm6', nn.BatchNorm2d(num_dec_features)),
            ('relu6', nn.ReLU(inplace=True)),
            ('final-conv2', nn.Conv2d(num_dec_features, 1, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

        self.decoder3 = nn.Sequential(OrderedDict([
            ('up3', nn.Upsample(scale_factor=2)),
            ('conv7', nn.Conv2d(num_features[4], num_dec_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm7', nn.BatchNorm2d(num_dec_features)),
            ('relu7', nn.ReLU(inplace=True)),
            ('final-conv3', nn.Conv2d(num_dec_features, 1, kernel_size=1, stride=1, padding=0, bias=True))
        ]))


    def forward(self, x):

        encoding1 = self.encoder1(x)
        print(encoding1.shape)
        encoding2 = self.encoder2(encoding1)
        print(encoding2.shape)
        encoding3 = self.encoder3(encoding2)
        print(encoding3.shape)

        decoding1 = self.decoder1(encoding1)
        decoding1 = F.relu(decoding1, inplace=True)
        decoding2 = self.decoder2(encoding2)
        decoding2 = F.relu(decoding2, inplace=True)
        decoding3 = self.decoder3(encoding3)
        decoding3 = F.relu(decoding3, inplace=True)

        return decoding1, decoding2, decoding3


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


