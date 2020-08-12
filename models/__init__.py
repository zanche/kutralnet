#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:26:58 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import os
import ast
import importlib
from torch import optim
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

from .libs.optim_nadam import Nadam
from .libs.losses import FocalLoss
from .libs.losses import ClassBalancedLoss


# activation functions
activations = dict()
activations['softmax'] = dict(
        fn= Softmax,
        params= dict(dim=1)
    )

activations['sigmoid'] = dict(
        fn= Sigmoid,
        params= dict() # default params
    )

# cost functions
losses = dict()
losses['ce'] = dict(
        fn= CrossEntropyLoss,
        params= dict()
    )

losses['bce'] = dict(
        fn= BCEWithLogitsLoss,
        params= dict()
    )

losses['focal'] = dict(
        fn= FocalLoss,
        params= dict()
    )

losses['cb'] = dict(
        fn= ClassBalancedLoss,
        params= dict()
    )

# models registered
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

models['efficientnet'] = dict(
        img_dims= (224, 224),
        model_name= 'EfficientNet',
        model_path= 'model_efficientnet.pth',
        class_name= 'EfficienNetWildFire',
        module_name= 'models.efficientnet',
        optimizer= optim.Adam,
        optimizer_params= dict(betas=(0.9, 0.99), 
                              weight_decay=1e-4),
        preprocess_train= 'resize',
        preprocess_val= 'resize',
        preprocess_test= 'resize',
        scheduler= optim.lr_scheduler.StepLR,
        scheduler_params= dict(step_size=30)
    )

models['fire_detection'] = dict(
        img_dims= (224, 224),
        model_name= 'FireDetection',
        model_path= 'model_fire_detection.pth',
        class_name= 'FireDetection',
        module_name= 'models.fire_detection',
        optimizer= optim.Adam,
        optimizer_params= dict(),
        preprocess_train= 'fire_detection',
        preprocess_val= 'fire_detection',
        preprocess_test= 'fire_detection',
        scheduler= None,
        scheduler_params= dict()
    )

def get_model_name(model_id):
    return models[model_id]['model_name']

def get_model_paths(models_root, model_name, dataset_name, version=None,
                    create_path=False):
    # main folder
    models_save_path = os.path.join(models_root, 'saved')
    # final folder for {model_name}/{dataset_name}/{version}
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
    
def get_model(model_id='kutralnet', num_classes=2, extra_params=dict()):
    model = None
    config = get_model_params(model_id)
    module = importlib.import_module(config['module_name'])        
    model = getattr(module, config['class_name'])
    params = {'classes': num_classes }

    if isinstance(extra_params, list):
        _model_params = dict()
        for _param in extra_params:
            key, val = _param.split("=")
            try:
                val = ast.literal_eval(val)                
            except:
                pass
            _model_params[key] = val
            
        params.update(_model_params)
        
    elif isinstance(extra_params, dict):
        params.update(extra_params)

    model = model(**params)

    return model
# end get_model


def get_loss(key_id='ce', extra_params=dict()):
    keys = key_id.split("_")
    no_keys = len(keys)
    
    if no_keys > 2:
        # loss_loss_activation pattern
        # recursive to make cb_*_activation
        act_id = keys[-1]
        # common extra params
        add_params = dict()
        # loss keys
        loss_idxs = list(range(no_keys -1))        
        loss_idxs.sort(reverse=True)        
        
        for idx in loss_idxs:
            loss_id = "{}_{}".format(keys[idx], act_id)
            loss_fn = get_loss(loss_id, extra_params=add_params)
            add_params.update(dict(loss_fn= loss_fn))
            add_params.update(extra_params)
        
        return loss_fn
        
    elif no_keys > 1:
        # loss_activation pattern
        loss_id, act_id = keys
    else:
        loss_id = key_id[0]
        act_id = None
    if not loss_id in losses:
        raise ValueError('Must choose a registered cost function', losses.keys())
        
    loss = losses[loss_id]
    params = loss['params']
    
    if loss_id in ['cb', 'focal'] and not act_id is None:
        params.update(dict(is_softmax= act_id == 'softmax'))
        
    params.update(extra_params)
    
    if len(params.keys()) > 0:
        # if params were passed
        return loss['fn'](**params)
    else:
        return loss['fn']()
# end get_loss

def get_activation(key_id='softmax', extra_params=None):
    """Load the activation function to estimate the probabilities."""
    keys = key_id.split("_")
    
    # lask key is activation
    act_id = keys[-1] if len(keys) > 1 else key_id
    
    if not act_id in activations:
        raise ValueError('Must choose a registered activation function', activations.keys())
        
    activation = activations[act_id]
    params = activation['params']
        
    if not extra_params is None:
        params.update(extra_params)
    
    if len(params.keys()) > 0:
        # if params were passed
        return activation['fn'](**params)
    else:
        return activation['fn']()
# end get_activation
