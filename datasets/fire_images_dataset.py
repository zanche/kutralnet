import os
import sys
from torchvision import transforms
sys.path.append('..')
from datasets.base import BaseDataset
from datasets.base import datasets_path


class FireImagesDataset(BaseDataset):
    def __init__(self, name, root_path, csv_file='dataset.csv', transform=None,
        purpose='train', preload=False):
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
        
        super().__init__(name=name, root_path=root_path, csv_file=csv_file, transform=transform,
            purpose=purpose, preload=preload)
        
    # end __init__    
# end FireImagesDataset

class FireNetDataset(FireImagesDataset):
    def __init__(self, transform=None, purpose='train', preload=False):
        super().__init__(name='FireNet', root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FireNetDataset

class FireNetTestDataset(FireImagesDataset):
    def __init__(self, transform=None, purpose='test', preload=False):
        super().__init__(name='FireNet-Test', root_path=os.path.join(datasets_path, 'FireNetDataset'),
            csv_file='dataset_test.csv', transform=transform, purpose=purpose, preload=preload)
    # end __init__
# end FireNetTestDataset

class CustomNormalize:
    def __init__(self, interval=(0, 1)):
        self.a = interval[0]
        self.b = interval[1]
    # end __init__

    def __call__(self, tensor):
        minimo = tensor.min()
        maximo = tensor.max()
        return (self.b - self.a) * ((tensor - minimo) / (maximo - minimo)) + self.a
    # end __call__

    def __repr__(self):
        return self.__class__.__name__ + '([{}, {}])'.format(self.a, self.b)
    # end __repr__
# end CustomNormalize

if __name__ == '__main__':
    data_path = os.path.join(datasets_path, 'FireNetDataset')

    print('data_path', data_path)

    transform_compose = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

    dataset = FireImagesDataset('FireNet', data_path, transform=transform_compose)
    print(dataset.data)
    sample = dataset[1618]
    print(sample)
    dataset.labels_describe()
