import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

datasets_path = os.path.dirname(os.path.abspath(__file__))

class BaseDataset(Dataset):
    def __init__(self, name, root_path, csv_file='dataset.csv', transform=None,
        purpose='train', preload=False):
        self.root_path = root_path
        self.csv_file = csv_file
        self.name = name
        self.purpose = purpose
        self.transform = transform
        self.preload = preload
        self.data = self.read_csv()
        
        if not hasattr(self, 'labels'):
            self.labels = None

        if self.preload:
            self._preload()
    # end __init__

    def __len__(self):
        return len(self.data)
    # end __len__

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload:
            return self.x[idx], self.y[idx]

        return self._item(idx)
    # end __getitem__

    def _item(self, idx):
        img_path = os.path.join(self.root_path,
                                self.data.iloc[idx]['folder_path'],
                                self.data.iloc[idx]['image_id'])
        image = Image.open(img_path).convert('RGB')
        # image = image
        label = self.data.iloc[idx]['class']
        label = self.labels[label]['idx']
        label = torch.from_numpy(np.array(label))

        if self.transform:
            image = self.transform(image)

        return image, label
    # end _item

    def read_csv(self):
        # csv read
        csv_path = os.path.join(self.root_path, self.csv_file)
        print('Reading {}({}) from file: {} (on-memory={})'.format(self.name,
                        self.purpose, csv_path, self.preload))
        dataset_df = pd.read_csv(csv_path)

        if self.purpose is not None and 'purpose' in dataset_df:
            dataset_df = dataset_df[dataset_df['purpose'] == self.purpose]

        return dataset_df
    # read_csv

    def _preload(self):
        self.x = []
        self.y = []
        
        since = time.time()
        for i in range(len(self.data)):
            item = self._item(i)
            self.x.append(item[0])
            self.y.append(item[1])
        time_elapsed = time.time() - since
        print('Dataset loaded in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # end _preload
# end BaseDataset
