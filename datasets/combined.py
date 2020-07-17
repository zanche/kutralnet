#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 03:08:29 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import os
import pandas as pd
from torchvision import transforms

from datasets.base import BaseDataset
from datasets.fire_smoke import FireNetV2Dataset
from datasets.fire_smoke import FireNetTestV2Dataset
from datasets.fire_smoke import FireFlameDataset
from datasets.fire_smoke import FireFlameTestDataset

class CombinedDataset(BaseDataset):
    
    def __init__(self, dataset1, dataset2, 
                 name=None, 
                 transform=None,
                 purpose='train', 
                 preload=False, 
                 one_hot=True, 
                 distributed=True, 
                 multi_label=True):
        self.dataset1 = dataset1(transform=transform, 
                                 purpose=purpose, 
                                 preload=False, 
                                 multi_label=multi_label)
        self.dataset2 = dataset2(transform=transform, 
                                 purpose=purpose, 
                                 preload=False, 
                                 multi_label=multi_label)
        root_path = os.path.commonpath([self.dataset1.root_path, 
                                        self.dataset2.root_path])
        if name is None:
            name = "{} and {} Dataset".format(self.dataset1.name, 
                                              self.dataset2.name)
        super(CombinedDataset, self).__init__(
                name=name, 
                root_path=root_path,
                transform=transform, 
                purpose=purpose, 
                preload=preload, 
                one_hot=one_hot, 
                distributed=distributed, 
                multi_label=multi_label)
    # end __init__
    
    def set_labels(self):
        labels1 = self.dataset1.labels
        labels1.update(self.dataset2.labels)
        self.labels = labels1
        self.label_default = self.dataset1.label_default        
        
    def read_csv(self):
        # csv read
        data1 = self.dataset1.data
        data2 = self.dataset2.data
        
        return pd.concat([data1, data2]).reset_index(drop=True)
    # read_csv
    
class FireFlameV2Dataset(CombinedDataset):    
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=True, 
                 distributed=True, 
                 multi_label=True):
        super(FireFlameV2Dataset, self).__init__(
            FireNetV2Dataset, FireFlameDataset,
            name='FireFlame v2', 
            transform=transform, 
            purpose=purpose,
            preload=preload, 
            one_hot=one_hot, 
            distributed=distributed, 
            multi_label=multi_label)
    
class FireFlameTestV2Dataset(CombinedDataset):    
    def __init__(self, 
                 transform=None, 
                 purpose='test', 
                 preload=False, 
                 one_hot=True, 
                 distributed=True, 
                 multi_label=True):
        super(FireFlameTestV2Dataset, self).__init__(
            FireNetTestV2Dataset, FireFlameTestDataset,
            name='FireFlame Test v2', 
            transform=transform, 
            purpose=purpose,
            preload=preload, 
            one_hot=one_hot, 
            distributed=distributed, 
            multi_label=multi_label)
        
if __name__ == '__main__':
    transform_compose = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    combined = FireFlameV2Dataset(transform=transform_compose, purpose=None)
    print(combined.data)
    print(len(combined))
    sample = combined[48]
    print('sample', sample[0].shape, sample[1])
    combined.labels_describe()
    combined.print_summary()
    
    combined = FireFlameTestV2Dataset(transform=transform_compose, purpose=None)
    print(combined.data)
    print(len(combined))
    sample = combined[48]
    print('sample', sample[0].shape, sample[1])
    combined.labels_describe()
    combined.print_summary()
