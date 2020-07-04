import os
from torchvision import transforms

from .base import BaseDataset
from .base import datasets_path


class FireImages(BaseDataset):
    def __init__(self, name, root_path, csv_file='dataset.csv', transform=None,
        purpose='train', preload=False):
        super(FireImages, self).__init__(name=name, 
                root_path=root_path, csv_file=csv_file, transform=transform,
                purpose=purpose, preload=preload)
        
    def set_labels(self):
        self.labels = {
            'none': {
                'idx': 0,
                'label': 'none',
                'name': 'NoFire'
            },
            'fire': {
                'idx': 1,
                'label': 'fire',
                'name': 'Fire'
            }
        }
        # default label
        self.set_label_default('none')
        
    # end __init__    
# end FireImagesDataset

class FireNetDataset(FireImages):
    def __init__(self, transform=None, purpose='train', 
                 preload=False, multi_label=True):
        super(FireNetDataset, self).__init__(name='FireNet', 
            root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset.csv', transform=transform, 
            purpose=purpose, preload=preload, multi_label=multi_label)
    # end __init__
# end FireNetDataset

class FireNetTestDataset(FireImages):
    def __init__(self, transform=None, purpose='test', 
                 preload=False, multi_label=True):
        super(FireNetTestDataset, self).__init__(name='FireNet Test', 
            root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset_test.csv', transform=transform, 
            purpose=purpose, preload=preload, multi_label=multi_label)
    # end __init__
# end FireNetTestDataset

class FiSmoDataset(FireImages):
    def __init__(self, transform=None, purpose='train', 
                 preload=False, multi_label=True):
        super(FiSmoDataset, self).__init__(name='FiSmo', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset.csv', transform=transform, 
            purpose=purpose, preload=preload, multi_label=multi_label)
    # end __init__
# end FiSmoDataset

class FiSmoBlackDataset(FireImages):
    def __init__(self, transform=None, purpose='train', 
                 preload=False, multi_label=True):
        super(FiSmoBlackDataset, self).__init__(name='FiSmoBlack', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_black.csv', transform=transform, 
            purpose=purpose, preload=preload, multi_label=multi_label)
    # end __init__
# end FiSmoBlackDataset

class FiSmoBalancedDataset(FireImages):
    def __init__(self, transform=None, purpose='train', 
                 preload=False, multi_label=True):
        super(FiSmoBalancedDataset, self).__init__(name='FiSmoBalanced', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced.csv', transform=transform, 
            purpose=purpose, preload=preload, multi_label=multi_label)
    # end __init__
# end FiSmoBalancedDataset

class FiSmoBalancedBlackDataset(FireImages):
    def __init__(self, transform=None, purpose='train', 
                 preload=False, multi_label=True):
        super(FiSmoBalancedBlackDataset, self).__init__(name='FiSmoBalancedBlack', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced_black.csv', transform=transform, 
            purpose=purpose, preload=preload, multi_label=multi_label)
    # end __init__
# end FiSmoBalancedBlackDataset

if __name__ == '__main__':
    data_path = os.path.join(datasets_path, 'FireNetDataset')

    print('data_path', data_path)

    transform_compose = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

    dataset = FireImages('FireNet', data_path, transform=transform_compose, purpose=None)
    print(dataset.data)
    sample = dataset[1618]
    print('sample', sample[0].shape, sample[1])
    dataset.labels_describe()
    
    # datasets summaries
    dataset = FireNetDataset(purpose=None) # get all images
    dataset.labels_describe()
    
    dataset = FireNetTestDataset(purpose=None) # get all images
    dataset.labels_describe()
    