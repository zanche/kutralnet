#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:49:51 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import os
from torchvision import transforms

from datasets.base import BaseDataset
from datasets.base import datasets_path

class FireSmokeImages(BaseDataset):
    def __init__(self, 
                 name, 
                 root_path, 
                 csv_file='dataset.csv', 
                 transform=None,
                 purpose='train', 
                 preload=False, 
                 one_hot=True, 
                 distributed=True, 
                 multi_label=True):
        super(FireSmokeImages, self).__init__(
                name=name, 
                root_path=root_path, 
                csv_file=csv_file, 
                transform=transform, 
                purpose=purpose, 
                preload=preload,
                one_hot=one_hot, 
                distributed=distributed, 
                multi_label=multi_label
            )
    # end __init__
    
    def set_labels(self):
        self.labels = {
            'none': {
                'idx': 0,
                'label': 'none',
                'name': 'None'
            },
            'fire': {
                'idx': 1,
                'label': 'fire',
                'name': 'Fire'
            },
            'smoke': {
                'idx': 2,
                'label': 'smoke',
                'name': 'Smoke'
            }
        }
        # concatenated classes for one-hot encode
        # if self.one_hot and self.multi_label:
        #     self.labels['fire_smoke'] = {
        #             'idx': self.labels['fire']['idx']
        #                     + self.labels['smoke']['idx'],
        #             'label': 'fire_smoke',
        #             'name': 'Fire & Smoke'
        #         }
    # end set_labels
# end FireSmokeImagesDataset

class FireNetV2Dataset(FireSmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=True,
                 distributed=True,
                 multi_label=True):
        super(FireNetV2Dataset, self).__init__(
            name='FireNet v2', 
            root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot,
            distributed=distributed,
            multi_label=multi_label)
    # end __init__
# end FireNetDataset

class FireNetTestV2Dataset(FireSmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='test', 
                 preload=False, 
                 one_hot=True,
                 distributed=True,
                 multi_label=True):
        super(FireNetTestV2Dataset, self).__init__(
            name='FireNet Test v2', 
            root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset_test_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot,
            distributed=distributed,
            multi_label=multi_label)
    # end __init__
# end FireNetTestDataset

class FiSmoV2Dataset(FireSmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=True,
                 distributed=True,
                 multi_label=True):
        super(FiSmoV2Dataset, self).__init__(
            name='FiSmo v2', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot,
            distributed=distributed,
            multi_label=multi_label)
    # end __init__
# end FiSmoDataset

class FiSmoBlackV2Dataset(FireSmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=True,
                 distributed=True,
                 multi_label=True):
        super(FiSmoBlackV2Dataset, self).__init__(
            name='FiSmoBlack v2', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_black_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot,
            distributed=distributed,
            multi_label=multi_label)
    # end __init__
# end FiSmoBlackDataset

class FiSmoBalancedV2Dataset(FireSmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=True,
                 distributed=True,
                 multi_label=True):
        raise NotImplementedError('The csv file must be complemented with smoke images.')
        super(FiSmoBalancedV2Dataset, self).__init__(
            name='FiSmoBalanced v2', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot,
            distributed=distributed,
            multi_label=multi_label)
    # end __init__
# end FiSmoBalancedDataset

class FiSmoBalancedBlackV2Dataset(FireSmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=True,
                 distributed=True,
                 multi_label=True):
        raise NotImplementedError('The csv file must be complemented with smoke images.')
        super(FiSmoBalancedBlackV2Dataset, self).__init__(
            name='FiSmoBalancedBlack v2', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced_black_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot,
            distributed=distributed,
            multi_label=multi_label)
    # end __init__
# end FiSmoBalancedBlackDataset

class FireFlameDataset(FireSmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=True,
                 distributed=True,
                 multi_label=True):
        super(FireFlameDataset, self).__init__(
            name='FireFlame', 
            root_path=os.path.join(datasets_path, 'FireFlameDataset'),
            csv_file='dataset.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot,
            distributed=distributed,
            multi_label=multi_label)
    # end __init__
# end FireFlameDataset

class FireFlameTestDataset(FireSmokeImages):
    def __init__(self, 
                 transform=None, 
                 purpose='test', 
                 preload=False, 
                 one_hot=True,
                 distributed=True,
                 multi_label=True):
        super(FireFlameTestDataset, self).__init__(
            name='FireFlame Test', 
            root_path=os.path.join(datasets_path, 'FireFlameDataset'),
            csv_file='dataset_test.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot,
            distributed=distributed,
            multi_label=multi_label)
    # end __init__
# end FireFlameDataset

if __name__ == '__main__':
    data_path = os.path.join(datasets_path, 'FiSmoDataset')

    print('data_path', data_path)

    transform_compose = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

    dataset = FireSmokeImages('FiSmo', data_path, transform=transform_compose, purpose=None)
    print(dataset.data)
    print(len(dataset))
    sample = dataset[48]
    print('sample', sample[0].shape, sample[1])
    dataset.labels_describe()
    
    # datasets summaries
    dataset = FireNetV2Dataset(purpose=None) # get all images
    if not dataset.is_splitted:
        dataset.split(persist=True)
    dataset.labels_describe()
    
    dataset = FireNetTestV2Dataset(purpose=None) # get all images
    dataset.labels_describe()
    
    dataset = FiSmoV2Dataset(purpose=None) # get all images
    if not dataset.is_splitted:
        dataset.split(persist=True)
    dataset.labels_describe()
    
    dataset = FiSmoBlackV2Dataset(purpose=None) # get all images
    if not dataset.is_splitted:
        dataset.split(persist=True)
    dataset.labels_describe()
    
    dataset = FireFlameDataset(purpose=None) # get all images
    if not dataset.is_splitted:
        dataset.split(persist=True)
    dataset.labels_describe()
    
    dataset = FireFlameTestDataset(purpose=None) # get all images
    dataset.labels_describe()
