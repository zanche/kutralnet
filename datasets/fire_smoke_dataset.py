#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:49:51 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import os
from torchvision import transforms

from .base import BaseDataset
from .base import datasets_path

class FireSmokeImagesDataset(BaseDataset):
    def __init__(self, name, root_path, csv_file='dataset.csv', transform=None,
        purpose='train', preload=False, hot_encode=True):
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
        
        super().__init__(name=name, root_path=root_path, csv_file=csv_file, 
                         transform=transform, purpose=purpose, preload=preload,
                         hot_encode=hot_encode)
        
    # end __init__
# end FireSmokeImagesDataset


class FiSmoDataset(FireSmokeImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        super().__init__(name='FiSmo', root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FiSmoDataset

class FiSmoBalancedDataset(FireSmokeImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        raise NotImplementedError('The csv file must be complemented with smoke images.')
        super().__init__(name='FiSmoBalanced', root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FiSmoBalancedDataset

class FiSmoBlackDataset(FireSmokeImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        raise NotImplementedError('The csv file must be complemented with smoke images.')
        super().__init__(name='FiSmoBlack', root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_black.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FiSmoBlackDataset

class FiSmoBalancedBlackDataset(FireSmokeImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        raise NotImplementedError('The csv file must be complemented with smoke images.')
        super().__init__(name='FiSmoBalancedBlack', root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced_black.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FiSmoBalancedBlackDataset

if __name__ == '__main__':
    data_path = os.path.join(datasets_path, 'FiSmoDataset')

    print('data_path', data_path)

    transform_compose = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

    dataset = FireSmokeImagesDataset('FiSmo', data_path, transform=transform_compose)
    print(dataset.data)
    print(len(dataset))
    sample = dataset[48]
    print(sample)
    