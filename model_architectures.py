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
        num_dec_features = [32, 64, 128]

        self.encoder1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_features[0], kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_features[0])),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(num_features[0], num_features[1], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(num_features[1])),
            ('relu1', nn.ReLU(inplace=True)),
        ]))

        self.encoder2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(num_features[1], num_features[2], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(num_features[2])),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

        self.encoder3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(num_features[2], num_features[3], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(num_features[3])),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(num_features[3], num_features[4], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(num_features[4])),
            ('relu4', nn.ReLU(inplace=True)),
        ]))

        self.decoder1 = nn.Sequential(OrderedDict([
            ('conv5', nn.ConvTranspose2d(num_features[1], num_dec_features[0], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm5', nn.BatchNorm2d(num_dec_features[0])),
            ('relu5', nn.ReLU(inplace=True)),
            ('final-conv1', nn.Conv2d(num_dec_features[0], 1, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

        self.decoder2 = nn.Sequential(OrderedDict([
            ('conv6', nn.ConvTranspose2d(num_features[2], num_dec_features[1], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm6', nn.BatchNorm2d(num_dec_features[1])),
            ('relu6', nn.ReLU(inplace=True)),
            ('final-conv2', nn.Conv2d(num_dec_features[1], 1, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

        self.decoder3 = nn.Sequential(OrderedDict([
            ('conv7', nn.ConvTranspose2d(num_features[4], num_dec_features[2], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm7', nn.BatchNorm2d(num_dec_features[2])),
            ('relu7', nn.ReLU(inplace=True)),
            ('final-conv3', nn.Conv2d(num_dec_features[2], 1, kernel_size=1, stride=1, padding=0, bias=True))
        ]))


    def forward(self, x):

        encoding1 = self.encoder1(x)
        encoding2 = self.encoder2(encoding1)
        encoding3 = self.encoder3(encoding2)

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


class SaliencyModelStandard(nn.Module):

    def __init__(self):

        super(SaliencyModelStandard, self).__init__()

        num_features = [48, 64, 96, 128, 256]
        num_dec_features = [128, 96, 64, 48]

        self.encoder = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_features[0], kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_features[0])),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(num_features[0], num_features[1], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(num_features[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2))
            ('conv2', nn.Conv2d(num_features[1], num_features[2], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(num_features[2])),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2))
            ('conv3', nn.Conv2d(num_features[2], num_features[3], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(num_features[3])),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=2))
            ('conv4', nn.Conv2d(num_features[3], num_features[4], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(num_features[4])),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2))
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('up1', nn.Upsample(scale_factor=2)),
            ('conv5', nn.Conv2d(num_features[4], num_dec_features[0], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm5', nn.BatchNorm2d(num_dec_features[0])),
            ('relu5', nn.ReLU(inplace=True)),
            ('up2', nn.Upsample(scale_factor=2)),
            ('conv6', nn.Conv2d(num_dec_features[0], num_dec_features[1], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm6', nn.BatchNorm2d(num_dec_features[1])),
            ('relu6', nn.ReLU(inplace=True)),
            ('up3', nn.Upsample(scale_factor=2)),
            ('conv7', nn.Conv2d(num_dec_features[1], num_dec_features[2], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm7', nn.BatchNorm2d(num_dec_features[2])),
            ('relu7', nn.ReLU(inplace=True)),
            ('up4', nn.Upsample(scale_factor=2)),
            ('conv8', nn.Conv2d(num_dec_features[2], num_dec_features[3], kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm8', nn.BatchNorm2d(num_dec_features[3])),
            ('relu8', nn.ReLU(inplace=True)),
            ('final-conv1', nn.Conv2d(num_dec_features[3], 1, kernel_size=1, stride=1, padding=0, bias=True))
        ]))


    def forward(self, x):

        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        decoding = F.relu(decoding, inplace=True)

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


class SaliencyModelHoles(nn.Module):

    def __init__(self):

        super(SaliencyModelHoles, self).__init__()

        num_features = [64, 64, 128, 256, 512]
        bn = 4000
        num_dec_features = [512, 256, 128, 64, 64]

        self.encoder = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_features[0], kernel_size=4, stride=2, padding=1, bias=False)),
            ('lrelu0', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('conv1', nn.Conv2d(num_features[0], num_features[1], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(num_features[1])),
            ('lrelu1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('conv2', nn.Conv2d(num_features[1], num_features[2], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(num_features[2])),
            ('lrelu2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('conv3', nn.Conv2d(num_features[2], num_features[3], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(num_features[3])),
            ('lrelu3', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('conv4', nn.Conv2d(num_features[3], num_features[4], kernel_size=4, stride=2, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(num_features[4])),
            ('lrelu4', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            ('conv5', nn.Conv2d(num_features[4], bn, kernel_size=4, stride=1, padding=0, bias=False)),
            ('norm5', nn.BatchNorm2d(bn)),
            ('lrelu5', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose2d(bn, num_dec_features[0], kernel_size=4, stride=1, padding=0, bias=False)),
            ('denorm1', nn.BatchNorm2d(num_dec_features[0])),
            ('derelu1', nn.ReLU(inplace=True)),
            ('deconv2', nn.ConvTranspose2d(num_dec_features[0], num_dec_features[1], kernel_size=4, stride=2, padding=1, bias=False)),
            ('denorm2', nn.BatchNorm2d(num_dec_features[1])),
            ('derelu2', nn.ReLU(inplace=True)),
            ('deconv3', nn.ConvTranspose2d(num_dec_features[1], num_dec_features[2], kernel_size=4, stride=2, padding=1, bias=False)),
            ('denorm3', nn.BatchNorm2d(num_dec_features[2])),
            ('derelu3', nn.ReLU(inplace=True)),
            ('deconv4', nn.ConvTranspose2d(num_dec_features[2], num_dec_features[3], kernel_size=4, stride=2, padding=1, bias=False)),
            ('denorm4', nn.BatchNorm2d(num_dec_features[3])),
            ('derelu3', nn.ReLU(inplace=True)),
            ('deconv5', nn.ConvTranspose2d(num_dec_features[3], num_dec_features[4], kernel_size=4, stride=2, padding=1, bias=False)),
            ('denorm5', nn.BatchNorm2d(num_dec_features[4])),
            ('derelu5', nn.ReLU(inplace=True)),
            ('deconv6', nn.ConvTranspose2d(num_dec_features[4], 1, kernel_size=4, stride=2, padding=1, bias=False)),
            ('derelu6', nn.ReLU(inplace=True))
        ]))


    def forward(self, x):

        out_temp = self.encoder(x)
        out = self.decoder(out_temp)
        return out


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)



