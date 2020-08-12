#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 01:42:41 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

from torch import nn
from efficientnet_pytorch import EfficientNet


def EfficienNetWildFire(classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')    
    num_ftrs = model._fc.in_features
    # modify the last layer to add another FC
    model._fc = nn.Linear(num_ftrs, classes)
    
    return model


if __name__ == '__main__':
    efficient = EfficienNetWildFire()
    print(efficient)