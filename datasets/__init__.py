# from .base import BaseDataset
from .utils import CustomNormalize
from .fire import FireNetDataset
from .fire import FireNetTestDataset
from .fire import FiSmoDataset
from .fire import FiSmoBalancedDataset
from .fire import FiSmoBlackDataset
from .fire import FiSmoBalancedBlackDataset
from .smoke import SmokeDataset
from .smoke import SmokeTestDataset
from .fire_smoke import FireNetV2Dataset
from .fire_smoke import FireNetTestV2Dataset
from .fire_smoke import FiSmoV2Dataset
from .fire_smoke import FiSmoBlackV2Dataset
# from .fire_smoke import FiSmoBalancedV2Dataset
# from .fire_smoke import FiSmoBalancedBlackV2Dataset
from .fire_smoke import FireFlameDataset
from .fire_smoke import FireFlameTestDataset
from .combined import FireFlameV2Dataset
from .combined import FireFlameTestV2Dataset
from .imagenet import ImageNetDataset


__all__ = [ 'FireNetDataset', 'FireNetTestDataset',
            'FiSmoDataset', 'FiSmoBlackDataset', 
            'FiSmoBalancedDataset', 'FiSmoBalancedBlackDataset',
            'ImageNetDataset', 'CustomNormalize',
            'FireNetV2Dataset', 'FireNetTestV2Dataset',
            'FiSmoV2Dataset', 'FiSmoBlackV2Dataset',
            # 'FiSmoBalancedV2Dataset', 'FiSmoBalancedBlackV2Dataset', 
            'FireFlameDataset', 'FireFlameTestDataset',
            'SmokeDataset', 'SmokeTestDataset',
            'FireFlameV2Dataset', 'FireFlameTestV2Dataset']

# registered datasets
datasets = dict()
datasets['firenet'] = {
        'name': 'FireNet',
        'class': FireNetDataset,
        'num_classes': 2
    }
datasets['firenet_test'] = {
        'name': 'FireNet Test',
        'class': FireNetTestDataset,
        'num_classes': 2
    }
datasets['fismo'] = {
        'name': 'FiSmo',
        'class': FiSmoDataset,
        'num_classes': 3
    }
datasets['fismo_black'] = {
        'name': 'FiSmoA',
        'class': FiSmoBlackDataset,
        'num_classes': 3
    }
datasets['fismo_balanced'] = {
        'name': 'FiSmoB',
        'class': FiSmoBalancedDataset,
        'num_classes': 3
    }
datasets['fismo_balanced_black'] = {
        'name': 'FiSmoBA',
        'class': FiSmoBalancedBlackDataset,
        'num_classes': 3
    }
datasets['imagenet']= {
        'name': 'ImageNet',
        'class': ImageNetDataset,
        'num_classes': 1000
    }
datasets['firenet_v2']= {
        'name': 'FireNet v2',
        'class': FireNetV2Dataset,
        'num_classes': 3
    }
datasets['firenet_testv2']= {
        'name': 'FireNet Test v2',
        'class': FireNetTestV2Dataset,
        'num_classes': 3
    }
datasets['fismo_v2']= {
        'name': 'FiSmo v2',
        'class': FiSmoV2Dataset,
        'num_classes': 3
    }
datasets['fismo_blackv2']= {
        'name': 'FiSmoA v2',
        'class': FiSmoBlackV2Dataset,
        'num_classes': 3
    }
datasets['fireflame']= {
        'name': 'FireFlame',
        'class': FireFlameDataset,
        'num_classes': 3
    }
datasets['fireflame_test']= {
        'name': 'FireFlame Test',
        'class': FireFlameTestDataset,
        'num_classes': 3
    }
datasets['smokeset']= {
        'name': 'SmokeSet',
        'class': SmokeDataset,
        'num_classes': 2
    }
datasets['smokeset_test']= {
        'name': 'SmokeSet Test',
        'class': SmokeTestDataset,
        'num_classes': 2
    }
datasets['fireflame_v2']= {
        'name': 'FireFlame v2',
        'class': FireFlameV2Dataset,
        'num_classes': 3
    }
datasets['fireflame_testv2']= {
        'name': 'FireFlame Test v2',
        'class': FireFlameTestV2Dataset,
        'num_classes': 3
    }
