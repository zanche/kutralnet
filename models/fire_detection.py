#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 02:46:04 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FireDetection(nn.Module):
    
    def __init__(self, classes, initial_filters=90):
        super(FireDetection, self).__init__()
        self._conv1 = nn.Conv2d(3, initial_filters, kernel_size=1, padding=1)
        self._pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self._conv2 = nn.Conv2d(90, 100, kernel_size=5, padding=3, stride=2)
        self._pool2 = nn.MaxPool2d(kernel_size=11, stride=2)
        
        self._conv3 = nn.Conv2d(100, 50, kernel_size=1)
        self._pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self._conv4 = nn.Conv2d(50, 50, kernel_size=3)
        self._conv5 = nn.Conv2d(50, 60, kernel_size=5)
        self._pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self._conv6 = nn.Conv2d(60, 10, kernel_size=2)
        
        self._fc1 = nn.Linear(40, 40)
        self._fc2 = nn.Linear(40, 40)
        self._fc3 = nn.Linear(40, 40)
        self._out = nn.Linear(40, classes)
        
    def forward(self, x):
        z = F.relu(self._conv1(x))
        z = self._pool1(z)
        z = F.relu(self._conv2(z))
        z = self._pool2(z)
        z = F.relu(self._conv3(z))
        z = self._pool3(z)
        z = F.relu(self._conv4(z))
        z = F.relu(self._conv5(z))
        z = self._pool4(z)
        z = F.relu(self._conv6(z))
        z = z.flatten(start_dim=1)
        z = self._fc1(z)
        z = self._fc2(z)
        z = self._fc3(z)
        y = self._out(z)
        return y
        