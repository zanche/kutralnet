import os
import ast
import time
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

datasets_path = os.path.dirname(os.path.abspath(__file__))

class BaseDataset(Dataset):
    def __init__(self, name, root_path, csv_file='dataset.csv', transform=None,
        purpose='train', preload=False, multi_label=False):
        self.root_path = root_path
        self.csv_file = csv_file
        self.name = name
        self.purpose = purpose
        self.transform = transform
        self.preload = preload
        self.data = self.read_csv()
        self.multi_label = multi_label
        
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
        # read image
        img_path = os.path.join(self.root_path,
                                self.data.iloc[idx]['folder_path'],
                                self.data.iloc[idx]['image_id'])
        image = Image.open(img_path).convert('RGB')
        
        # read label        
        label = self.data.iloc[idx]['class']
        label = self.label2tensor(label)
        
        # apply transformations
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
            dataset_df.reset_index(inplace=True)
            del dataset_df['index']

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
    
    def label2idx(self, label):
        idx = []
        
        if '[' in label: # is a list
            label = ast.literal_eval(label)
            for l in label:
                if l in self.labels:
                    idx.append(self.labels[l]['idx'])
        else:
            if label in self.labels:
                idx.append(self.labels[label]['idx'])
            else:
                idx.append(label)
                
        if (not self.multi_label
            and len(idx) == 1):
            return idx[0]
        else:
            return idx     
    # end label2idx
    
    def label2tensor(self, label):
        label_id = self.label2idx(label)
        
        
        if isinstance(label_id, str):
            print('Warning: no label \'{}\' matches!'.format(label_id),
                  'first label \'{}\' used instead'.format(next(iter(self.labels))))
            label_id = 0
        elif (isinstance(label_id, list) 
            and len(label_id) == 0):
            print('Warning: no label matches! first',
                  'label \'{}\' used instead'.format(next(iter(self.labels))))
            label_id = 0
        
        # to tensor            
        labels_tensor = torch.as_tensor(label_id)
        
        if self.multi_label:
            # Create one-hot encodings of labels
            one_hot = torch.nn.functional.one_hot(labels_tensor, 
                                                  num_classes=len(self.labels))
            # if multi-label
            return torch.sum(one_hot, dim=0).float()
        # print(labels_tensor)
        return labels_tensor
    # end label2tensor
    
    def labels_describe(self):
        print(self.data[
            ['image_id', 'class']].groupby(['class']).agg(['count']))
           
# end BaseDataset
