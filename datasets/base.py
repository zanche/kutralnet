import os
import ast
import math
import time
import torch
import collections
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

datasets_path = os.path.dirname(os.path.abspath(__file__))

class BaseDataset(Dataset):
    """
    The main class to use the fire, smoke and fire&smoke datasets.
    
    This is a generic class to use a dataset with the PyTorch's DataLoader.
    This class manage the instances of a dataset stored in a CSV file.
    Process each indexed element to apply the transformations, filter by 
    training, validation or testing purpose.
    Can load all the images on-memory before their use, this can speed up the
    training time, but with a high memory consumption, useful for small datasets.
    Process the label to give a Tensor, optionally can be one-hot encoded and
    use a distributed representation.
    If the image is multi-label, the labels can be concatenated as a new label,
    distributed or not, or can be excluded from the dataset.
    """
    
    def __init__(self, name, root_path, csv_file='dataset.csv', transform=None,
        purpose='train', preload=False, one_hot=False, distributed=False, 
        multi_label=True):
        """Read a CSV file containing the images path, id, class and purpose.

        Parameters
        ----------
        name string:
            The name for the dataset, just for printing purpose.
        root_path string or os.path: 
            The root path containing the CSV file and the images.
        csv_file string:
            The CSV filename. The default is 'dataset.csv'.
        transform torchvision.transform: optional
            The transformation to be applied in the image. The default is None.
        purpose string: optional
            The purpose filter value contained in the CSV. The default is 'train'.
        preload bool: optional
            If must or not load the images on-memory. The default is False.
        one_hot bool: optional
            If must or not one-hot encode the label. The default is False.
        distributed bool: optional
            If True, this dataset is intented to be used with a distributed 
            representation, excluding the first label because no contain any 
            class of the classification problem. The default is False.
        multi_label bool: optional
            If False, exclude images with more than one label from dataset.
            The default is True.
        """
        self.root_path = root_path
        self.csv_file = csv_file
        self.name = name
        self.purpose = purpose
        self.transform = transform
        self.preload = preload
        self.one_hot = one_hot
        self.distributed = distributed
        self.multi_label = multi_label
        self.data = self.read_csv()
        self.loaded = False
        self.labels_missing = []
        
        if not hasattr(self, 'labels'):
            self.set_labels()
            
        if not hasattr(self, 'label_default'):
            self.set_label_default()
        
        if self.preload:
            self.load_data()
            
    # end __init__
    
    @property
    def is_splitted(self):
        """Indicate if the dataset have different instances purpose of use."""
        if not 'purpose' in self.data:
            return False
        
        splits = self.data['purpose'].value_counts()
        return len(splits.keys()) > 1
    # end is_splitted
    
    @property
    def num_classes(self):
        num_classes = len(self.labels)
        
        if (self.one_hot and 
            (self.distributed
             or not self.multi_label)):
            num_classes -= 1
        
        return num_classes

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)
    # end __len__

    def __getitem__(self, idx):
        """Get a desired pair of (image, label) by index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.loaded:
            return self.x[idx], self.y[idx]

        return self._item(idx)
    # end __getitem__

    def _item(self, idx):
        """
        Get the (image, label) pair by index.
        
        A helper method to obtain a pair, if was preloaded, return from 
        memory, if not read and return the pair.        
        """
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
        """Set the labels from the dataset.
        
        Due this is a generic class, whe use or inherit the class, the posible
        labels mus be defined.
        
        To achieve this the class must set the labels attribute as a dict.
        The dictionary must contain the next structure:
        self.labels = {
            'class_0':{ -> this is the label string contained in the class column of the CSV
                'idx': 0, -> the int value for the label
                'label': 'class_0', -> the string value for the label
                'name': 'Clase 0' -> the human readable string for the label
                },
            ...
            }
        Additionally can be called the set_label_default method to assign a
        default label in case when the CSV class column contain a label not 
        specified in self.labels
        """
        raise NotImplementedError('Labels must be specified before use.')
    # end set_labels
        
    def set_label_default(self, label=None):
        """Set the default label.

        Parameters
        ----------
        label string: optional
            The self.labels key to be assigned as default. The default is None.
            If None is specified, the first label is considered as default.
        """
        # first label is used
        if label is None:
            label = next(iter(self.labels))
        
        self.label_default = self.labels[label]
        return self.label_default
    # end set_label_default

    def read_csv(self):
        """Read the CSV file into a pandas.DataFrame."""
        # csv read
        csv_path = os.path.join(self.root_path, self.csv_file)
        print_purpose = 'all' if self.purpose is None else self.purpose
        print('Reading {}({}) from file: {}'.format(self.name,
                        print_purpose, 
                        str(os.path.sep).join(csv_path.split(os.path.sep)[-3:]))
              )
        if self.one_hot:
            print("Using one-hot encoded labels")
        
        if self.distributed:
            print("Using distributed representation")
        
        dataset_df = pd.read_csv(csv_path)
        # append base folder path
        dataset_df['base_path'] = dataset_df.apply(lambda row: self.root_path, axis=1)

        # filter by purpose use
        if self.purpose is not None and 'purpose' in dataset_df:
            dataset_df = dataset_df[dataset_df['purpose'] == self.purpose]
            dataset_df = dataset_df.reset_index(drop=True)
            
        # filter multi-label images
        if not self.multi_label:
            print("Multi-label images are skip.")
            dataset_df = dataset_df[~dataset_df['class'].str.contains("\[")]
            dataset_df = dataset_df.reset_index(drop=True)

        return dataset_df
    # read_csv

    def load_data(self):
        """Load the data on-memroy for it use."""
        self.x = []
        self.y = []
        
        print('Loading dataset...')
        since = time.time()
        for i in range(len(self.data)):
            item = self._item(i)
            self.x.append(item[0])
            self.y.append(item[1])
        time_elapsed = time.time() - since
        print('{} images loaded in {:.0f}m {:.0f}s'.format(
            len(self), time_elapsed // 60, time_elapsed % 60))
        self.loaded = True
        
        # print summary
        samples = self.samples_by_class
        print('Labels:', end=' ')
        
        for k, label in self.labels.items():
            print("{}: {}".format(label['idx'], label['name']), end=", ")
        print()
        
        for k, smpl in samples.items():
            print("{}: {}\t({:.2f}%)".format(k, smpl['n'], smpl['p'] * 100.))
            
    # end _preload
    
    def label2idx(self, label):
        """Encode the label key into the int value.
        
        If no label is found return the same value as input.
        Can process list, and string values contained in the pandas.DataFrame.
        """
        idx = []
        
        def getValidId(label):
            if not label in self.labels:
                notice = 'Warning: label \'{}\' no registered! '.format(label)
                notice += 'default label \'{}\' is used instead.'.format(
                      self.label_default['label'])
                
                if not label in self.labels_missing:
                    print(notice)
                    self.labels_missing.append(label)
                
                return self.label_default['idx']
            
            return self.labels[label]['idx']
        
        if '[' in label: # is a list
            label = ast.literal_eval(label)
            for l in label:
                idx.append(getValidId(l))
        else:
            idx.append(getValidId(label))
        
        if self.one_hot:
            # if must one-hot encode
            if not self.distributed and len(idx) > 1:
                # if not distributed multi-label
                return [sum(idx)]
            else:
                # if single bit one-hot encode
                return idx
        else:
            # if class number
            return idx[0]
                
    # end label2idx
    
    def label2tensor(self, label):
        """Transform the label into a Tensor.
        
        First encode the instances label as int index.
        When the label is not in the dataset labels, the default label is used 
        instead.        
        If the one_hot attribute is True, the label is one-hot encoded.
        If the exclude_none attribute is True, the first column/values of the 
        one-hot encoded label is removed.
        """
        label_id = self.label2idx(label)
        
        # to tensor            
        labels_tensor = torch.as_tensor(label_id)
        
        if self.one_hot:
            # Create one-hot encodings of labels
            # print(self.num_classes, len(self.labels))
            one_hot = torch.nn.functional.one_hot(labels_tensor, 
                                                  num_classes=len(self.labels))
            
            # if distributed first label must be excluded.
            if self.distributed:
                one_hot = one_hot[:, 1:]
                
            #reduce dimensions
            one_hot = torch.sum(one_hot, dim=0).float()
            # in case is multi-label
            labels_tensor = one_hot
        
        return labels_tensor
    # end label2tensor
    
    def labels_describe(self, full=False):
        """Print the classes summary of the dataset's CSV.

        Parameters
        ----------
        full bool: optional
            Set to True if want to include the purpose on the summary. 
            The default is False.
        """
        if full:
            cols = ['image_id', 'class', 'purpose']
            groups = ['class', 'purpose']
        else:
            cols = ['image_id', 'class']
            groups = ['class']
            
        print(self.data[cols].groupby(groups).agg(['count']))
    # end labels_describe
    
    @property
    def samples_by_class(self):
        """Probality for each class in the dataset."""
        labels = dict()
        
        for idx in range(len(self.data)):
            # read label        
            label = self.data.iloc[idx]['class']
            label_id = str(self.label2idx(label))
            
            if not label_id in labels:
                labels[label_id] = 0
                
            labels[label_id] += 1
            
        labels = collections.OrderedDict(sorted(labels.items()))
        samples = dict()
        
        for k in labels.keys():
            n = labels[k]
            p = labels[k] / len(self.data)
            samples[k] = dict(n=n, p=p)
            
        return samples
    
    def split(self, size=0.2, persist=False):
        """Split the dataset into training and validation at specified size.

        Parameters
        ----------
        size float: optional
            The percetage of split, i.e., if 0.2 is used, the dataset will be
            splitted in 0.8 as training adn 0.2 as validation. 
            The default is 0.2.
        persist bool: optional
            Set to True if want save the changes into the CSV file. 
            The default is False.
        """
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
    # end split
        
    def __str__(self):
        """Representate as string."""
        # name = "{}(csv_file={}, purpose={}, one_hot={}, distributed={}, multi_label={})".format(
        #     self.name, self.csv_file, self.purpose, 
        #     self.one_hot, self.distributed, self.multi_label)
        return self.name
    # end __str__
           
# end BaseDataset
        