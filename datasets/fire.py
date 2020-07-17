import os
from torchvision import transforms

from datasets.base import BaseDataset
from datasets.base import datasets_path


class FireImages(BaseDataset):
    def __init__(self, 
                 name, 
                 root_path, 
                 csv_file='dataset.csv', 
                 transform=None,
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FireImages, self).__init__(
                name=name, 
                root_path=root_path, 
                csv_file=csv_file, 
                transform=transform,
                purpose=purpose, 
                preload=preload, 
                one_hot=one_hot
            )
    # end __init__
        
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
    # end set_labels    
# end FireImagesDataset

class FireNetDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FireNetDataset, self).__init__(
            name='FireNet', 
            root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FireNetDataset

class FireNetTestDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='test', 
                 preload=False, 
                 one_hot=False):
        super(FireNetTestDataset, self).__init__(
            name='FireNet Test', 
            root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset_test.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FireNetTestDataset

class FiSmoDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FiSmoDataset, self).__init__(
            name='FiSmo', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FiSmoDataset

class FiSmoBlackDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FiSmoBlackDataset, self).__init__(
            name='FiSmoBlack', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_black.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FiSmoBlackDataset

class FiSmoBalancedDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FiSmoBalancedDataset, self).__init__(
            name='FiSmoBalanced', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FiSmoBalancedDataset

class FiSmoBalancedBlackDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FiSmoBalancedBlackDataset, self).__init__(
            name='FiSmoBalancedBlack', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced_black.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FiSmoBalancedBlackDataset


## relabeled versions
class FireNetRDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FireNetRDataset, self).__init__(
            name='FireNet-R', 
            root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FireNetRDataset

class FireNetTestRDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='test', 
                 preload=False, 
                 one_hot=False):
        super(FireNetTestRDataset, self).__init__(
            name='FireNet Test-R', 
            root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset_test_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FireNetTestRDataset

class FiSmoRDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FiSmoRDataset, self).__init__(
            name='FiSmo-R', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FiSmoRDataset

class FiSmoBlackRDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FiSmoBlackRDataset, self).__init__(
            name='FiSmoBlack-R', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_black.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FiSmoBlackRDataset

class FiSmoBalancedRDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FiSmoBalancedRDataset, self).__init__(
            name='FiSmoBalanced-R', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FiSmoBalancedRDataset

class FiSmoBalancedBlackRDataset(FireImages):
    def __init__(self, 
                 transform=None, 
                 purpose='train', 
                 preload=False, 
                 one_hot=False):
        super(FiSmoBalancedBlackRDataset, self).__init__(
            name='FiSmoBalancedBlack-R', 
            root_path=os.path.join(datasets_path, 'FiSmoDataset'),
            csv_file='dataset_balanced_black_v2.csv', 
            transform=transform, 
            purpose=purpose, 
            preload=preload, 
            one_hot=one_hot)
    # end __init__
# end FiSmoBalancedBlackDataset

if __name__ == '__main__':
    data_path = os.path.join(datasets_path, 'FireNetDataset')
    print('data_path', data_path)

    # generic form fire datasets
    dataset = FireImages('FireNet', data_path, purpose=None)
    sample = dataset[1618]
    print('sample', sample[0].shape, sample[1])
    dataset.labels_describe()
    
    # datasets summaries
    dataset = FireNetDataset(purpose=None) # get all images
    print(dataset)
    dataset.labels_describe()
    
    dataset = FireNetTestDataset(purpose=None) # get all images
    print(dataset)
    dataset.labels_describe()
    