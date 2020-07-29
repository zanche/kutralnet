#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:11:17 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

from datasets import datasets
from datasets import get_dataset
import pandas as pd

if __name__ == '__main__':
    for ds in datasets.keys():
        if ds in ['imagenet', 'fismo_balanced_relabeled', 'fismo_balanced_black_relabeled']:
            continue
        d = get_dataset(ds, params=dict(purpose=None))
        d.labels_describe(True)
        samples_class = d.samples_by_class        
        for k, sample in samples_class.items():
            print(k, sample['n'], "{:.4f}".format(sample['p']))
        print()
    