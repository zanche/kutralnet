#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 01:32:12 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

from torchvision import transforms


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

class ImagePreprocess(transforms.Compose):
    """Standard pre-process transformation for the images."""
    
    def __init__(self, img_dims=(224, 224)):
        """Set the resize transform."""
        super(ImagePreprocess, self).__init__([
                        transforms.Resize(img_dims), #redimension
                        transforms.ToTensor()
                    ])
    # end __init__
# end ImagePreprocess

        

