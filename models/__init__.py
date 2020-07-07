#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:26:58 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import os
import importlib
from torch import optim
from torch.nn import Sigmoid
from torch.nn import LogSoftmax
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

from .libs.optim_nadam import Nadam


# activations and cost function
activations = dict()

activations['softmax'] = dict(
        loss= CrossEntropyLoss,
        loss_params= None,
        activation= LogSoftmax,
        activation_params= dict(dim=1)
    )

activations['sigmoid'] = dict(
        loss= BCEWithLogitsLoss,
        loss_params= None,
        activation= Sigmoid,
        activation_params= None
    )

models = dict()
models['firenet'] = dict(
        img_dims= (64, 64),
        model_name= 'FireNet',
        model_path= 'model_firenet.pth',
        class_name= 'FireNet',
        module_name= 'models.firenet_pt',
        optimizer= optim.Adam,
        optimizer_params= dict(eps= 1e-6),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

models['firenet_tf'] = dict(
        img_dims= (64, 64),
        model_name= 'FireNet',
        model_path= 'model_firenet_tf.h5',
        class_name= 'firenet_tf',
        module_name= 'models.firenet_tf',
    )

models['octfiresnet'] = dict(
        img_dims= (96, 96),
        model_name= 'OctFiResNet',
        model_path= 'model_octfiresnet.pth',
        class_name= 'OctFiResNet',
        module_name= 'models.octfiresnet',
        optimizer= Nadam,
        optimizer_params= dict(lr= 1e-4, eps= 1e-7),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

models['resnet'] = dict(
        img_dims= (224, 224),
        model_name= 'ResNet50',
        model_path= 'model_resnet.pth',
        class_name= 'resnet_sharma',
        module_name= 'models.resnet',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

models['kutralnet'] = dict(
        img_dims= (84, 84),
        model_name= 'KutralNet',
        model_path= 'model_kutralnet.pth',
        class_name= 'KutralNet',
        module_name= 'models.kutralnet',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= optim.lr_scheduler.StepLR,
        scheduler_params= dict(step_size=85)
    )

models['kutralnetoct'] = dict(
        img_dims= (84, 84),
        model_name= 'KutralNet Octave',
        model_path= 'model_kutralnetoct.pth',
        class_name= 'KutralNetOct',
        module_name= 'models.kutralnetoct',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

models['kutralnet_mobile'] =  dict(
        img_dims= (84, 84),
        model_name= 'KutralNet Mobile',
        model_path= 'model_kutralnet_mobile.pth',
        class_name= 'KutralNetMobile',
        module_name= 'models.kutralnet_mobile',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

models['kutralnet_mobileoct'] = dict(
        img_dims= (84, 84),
        model_name= 'KutralNet Mobile Octave',
        model_path= 'model_kutralnet_mobileoct.pth',
        class_name= 'KutralNetMobileOct',
        module_name= 'models.kutralnet_mobileoct',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

models['kutralnet_pre'] = dict(
        img_dims= (84, 84),
        model_name= 'KutralNet Pretrained',
        model_path= 'model_kutralnet.pth',
        class_name= 'KutralNetPreTrained',
        module_name= 'models.kutralnet',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

models['kutralnet_mobileoct_pre'] = dict(
        img_dims= (84, 84),
        model_name= 'KutralNet Mobile Octave Pretrained',
        model_path= 'model_kutralnet_mobileoct.pth',
        class_name= 'KutralNetMobileOctPreTrained',
        module_name= 'models.kutralnet_mobileoct',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

models['resnet18'] = dict(
        img_dims= (224, 224),
        model_name= 'ResNet18',
        model_path= 'model_resnet18.pth',
        class_name= 'ResNet18',
        module_name= 'models.resnet',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= None,
        scheduler_params= dict()
    )

def get_model_name(model_id):
    return models[model_id]['model_name']

def get_model_paths(models_root, model_name, dataset_name, version=None,
                    create_path=False):
    # main folder
    models_save_path = os.path.join(models_root, 'saved')
    # final folder for {model_name}/{dataset_name}_{version}
    folder_name = os.path.join(model_name, dataset_name)
    if version is not None:
        folder_name = os.path.join(folder_name, version)
        
    save_path = os.path.join(models_save_path, folder_name)
    save_path = os.path.abspath(save_path)
    
    model_path = os.path.join(save_path, models[model_name]['model_path'])
    model_path = os.path.abspath(model_path)
    
    # if must create
    if create_path:
        # models root
        if not os.path.exists(models_root):
            print('Creating models_root path', models_root)
            os.makedirs(models_root)
        
        # model save path folder
        if not os.path.exists(save_path):
            print('Creating save path', save_path)
            os.makedirs(save_path)
    
    else:
        # dir exists?
        if not os.path.isdir(save_path):
            save_path, model_path = None, None
            
    return save_path, model_path
# end get_paths

def get_model_params(model_id='kutralnet'):
    params = None
    if model_id in models:
        params = models[model_id]        
    else:
        raise ValueError('Must choose a registered model', models.keys())
    
    return params
    
def get_model(model_id='kutralnet', num_classes=2, extra_params=None):
    model = None
    config = get_model_params(model_id)
    module = importlib.import_module(config['module_name'])        
    model = getattr(module, config['class_name'])
    params = {'classes': num_classes }

    if not extra_params is None:
        params.update(extra_params)

    model = model(**params)

    return model
# end get_model


def get_loss(act_id='softmax'):
    if not act_id in activations:
        raise ValueError('Must choose a registered cost function', activations.keys())
        
    loss_class = activations[act_id]['loss']
    params = activations[act_id]['loss_params']
    
    if not params is None:
        return loss_class(**params)
    else:
        return loss_class()
# end get_loss

def get_activation(act_id='softmax'):
    if not act_id in activations:
        raise ValueError('Must choose a registered activation function', activations.keys())
        
    activation_class = activations[act_id]['activation']
    params = activations[act_id]['activation_params']
    
    if not params is None:
        return activation_class(**params)
    else:
        return activation_class()
# end get_activation
