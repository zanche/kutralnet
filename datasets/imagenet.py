import os
import sys

from datasets.base import BaseDataset
from datasets.base import datasets_path
from torchvision import transforms

class ImageNetDataset(BaseDataset):
    def __init__(self, csv_file='dataset.csv', transform=None,
        purpose='train', preload=False):
        super().__init__(name='ImageNet', root_path=os.path.join(datasets_path, 'ImageNetDataset'),
            csv_file=csv_file, transform=transform, purpose=purpose, preload=preload)
    # end __init__

    def set_labels(self):
        print('Processing labels...')
        labels = dict()
        for idx, wnid in enumerate(self.data['class'].unique()):
            label = {
                'idx': idx,
                'label': wnid,
                'name': self.data[self.data['class'] == wnid].iloc[0]['class_name']
            }
            labels.update({wnid: label})
        print("{} labels ok".format(len(labels)))
        self.labels = labels
    # end get_labels
# end ImageNetDataset

if __name__ == '__main__':
    transform_compose = transforms.Compose([
                transforms.Resize((86, 86)),
                transforms.ToTensor()
            ])

    dataset = ImageNetDataset(transform=transform_compose)
    #print(dataset.labels)
    print(len(dataset))
    sample = dataset[1618]
    print(sample)
