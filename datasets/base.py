import os
import ast
import math
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
        self.multi_label = multi_label
        self.data = self.read_csv()
        self.loaded = False
        
        if not hasattr(self, 'labels'):
            self.set_labels()
            
        if not hasattr(self, 'label_default'):
            self.set_label_default()
        
        if self.preload:
            self.load_data()
            
    # end __init__
    
    @property
    def is_splitted(self):
        if not 'purpose' in self.data:
            return False
        
        splits = self.data['purpose'].value_counts()
        return len(splits.keys()) > 1

    def __len__(self):
        return len(self.data)
    # end __len__

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.loaded:
            return self.x[idx], self.y[idx]

        return self._item(idx)
    # end __getitem__

    def _item(self, idx):
        # read image
        img_path = os.path.join(self.data.iloc[idx]['base_path'],
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
        
    def set_labels(self):
        raise NotImplementedError('Labels must be specified before use.')
        
    def set_label_default(self, label=None):
        # first label is used
        if label is None:
            label = next(iter(self.labels))
        
        self.label_default = self.labels[label]
        return self.label_default

    def read_csv(self):
        # csv read
        csv_path = os.path.join(self.root_path, self.csv_file)
        print_purpose = 'all' if self.purpose is None else self.purpose
        print('Reading {}({}) from file: {}'.format(self.name,
                        print_purpose, 
                        str(os.path.sep).join(csv_path.split(os.path.sep)[-3:]))
              )
        
        dataset_df = pd.read_csv(csv_path)
        # append base folder path
        dataset_df['base_path'] = dataset_df.apply(lambda row: self.root_path, axis=1)

        if self.purpose is not None and 'purpose' in dataset_df:
            dataset_df = dataset_df[dataset_df['purpose'] == self.purpose]
            dataset_df.reset_index(inplace=True)
            del dataset_df['index']

        return dataset_df
    # read_csv

    def load_data(self):
        self.x = []
        self.y = []
        
        print('Loading dataset...')
        since = time.time()
        for i in range(len(self.data)):
            item = self._item(i)
            self.x.append(item[0])
            self.y.append(item[1])
        time_elapsed = time.time() - since
        print('Dataset loaded in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        self.loaded = True
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
        
        def isValidLabel(label):
            if isinstance(label_id, str) or label is None:
                print('Warning: no label \'{}\' registered!'.format(label_id),
                  'default label \'{}\' used instead'.format(
                      self.label_default['label']))
                return False
            return True
        
        # check labels
        if self.multi_label:
            for i, l in enumerate(label_id):
                if not isValidLabel(l):
                    label_id[i] = self.label_default['idx']
        else:
            if not isValidLabel(label_id):
                label_id = self.label_default['idx']           
        
        # to tensor            
        labels_tensor = torch.as_tensor(label_id)
        
        if self.multi_label:
            # Create one-hot encodings of labels
            one_hot = torch.nn.functional.one_hot(labels_tensor, 
                                                  num_classes=len(self.labels))
            # if multi-label
            return torch.sum(one_hot, dim=0).float()
        
        return labels_tensor
    # end label2tensor
    
    def labels_describe(self, full=False):
        if full:
            cols = ['image_id', 'class', 'purpose']
            groups = ['class', 'purpose']
        else:
            cols = ['image_id', 'class']
            groups = ['class']
            
        print(self.data[cols].groupby(groups).agg(['count']))
    # end labels_describe
    
    
    def split(self, size=0.2, persist=False):
        # classes count
        instance_by_label = self.data['class'].value_counts()
        split_df = None
        
        print('Splitting dataset...')        
        for label in instance_by_label.keys():
            # calculate number of elements by label
            n_images = math.ceil(instance_by_label[label] * size)
            
            # filter by label and split
            label_df = self.data.loc[self.data['class'] == label]
            
            # shufle data and reset
            label_df = label_df.sample(frac=1.)
            label_df = label_df.reset_index(drop=True)
            
            # purpose labeling
            label_df['purpose'] = 'train'
            label_df.loc[label_df.tail(n_images).index, 'purpose'] = 'val'
            
            # concat splitted dataset
            split_df = pd.concat([split_df, label_df])
            
        # order data reset
        split_df.sort_values(['folder_path', 'image_id'], inplace=True)
        split_df.reset_index(drop=True, inplace=True)
        # summary
        print(split_df.groupby(['class', 'purpose']).agg({'purpose': ['count']}))
        
        if persist:
            print('Saving changes...')
            split_df.to_csv(os.path.join(self.root_path, self.csv_file), 
                            index=False)
    
        self.data = split_df
           
# end BaseDataset
        