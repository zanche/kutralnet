#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 00:44:26 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """Focal loss cost function implementation.
    
    Based on implementatios by:
    [github/CoinCheung](https://github.com/CoinCheung/pytorch-loss/blob/master/focal_loss.py)
    and 
    [github/vandit15](https://github.com/vandit15/Class-balanced-loss-pytorch)
    # version 1: use torch.autograd
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean', is_softmax=False):
        """Initialize parameters for Focal loss."""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction
        self.weight = None
        self.is_softmax = bool(is_softmax)
        
        if is_softmax:
            self.crit = nn.CrossEntropyLoss(reduction='none')
        else:
            self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        """Compute Focal loss for the input.
        
        Args:
            logits Tensor of shape (N, ...): input predicted logits values
            target Tensor of shape(N, ...): the the y values for the input
        """
        # if not hot_encoded
        if target.dim() < 2:
            target = F.one_hot(target, input.size(-1)).float()

        # compute loss
        logits = input.float() # use fp32 if logits is fp16
        
        if not self.alpha is None:
            # calculate weights
            with torch.no_grad():
                self.weight = torch.empty_like(logits).fill_(1 - self.alpha)
                self.weight[target == 1] = self.alpha
            
        # numeric transformation of probalities for numeric stability
        if self.is_softmax:
            # softmax activation
            probabilities = -F.log_softmax(logits, dim=1)
            labels = torch.argmax(target, dim=1)
        else:
            # sigmoid activation
            probabilities = -F.logsigmoid(logits)
            labels = target
            
        cross_entropy = self.crit(logits, labels)
        
        # probs = torch.sigmoid(logits)
        # pt = torch.where(target == 1, probs, 1 - probs)
        # modulator = torch.pow(1.0 - pt, self.gamma)
        
        # A numerically stable implementation of modulator.
        if self.gamma == 0.0:
            modulator = 1.0
        else:            
            modulator = torch.exp(-self.gamma * target * logits 
                                  -self.gamma * probabilities)
            
        if self.is_softmax:
            # extend values to batch size fit
            cross_entropy = cross_entropy.unsqueeze(1)
            cross_entropy = cross_entropy.repeat(1, logits.size(-1))
            
        loss = modulator * cross_entropy
        print('modulator', modulator.shape, 
              'cross_entropy', cross_entropy.shape,
              'self.weight', self.weight.shape)

        # weighted loss
        if not self.weight is None:
            loss = self.weight * loss
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss
    
        # focal_loss = torch.sum(weighted_loss)
        # # Normalize by the total number of positive samples.
        # focal_loss /= torch.sum(target)
        # return focal_loss

class ClassBalancedLoss(nn.Module):    
    """Class balanced loss cost function implementation.
    
    Based on implementatios by 
    [github/vandit15](https://github.com/vandit15/Class-balanced-loss-pytorch)
    """
    
    def __init__(self, samples_per_cls, loss_fn=nn.BCEWithLogitsLoss(), 
                 beta=0.9999, is_softmax=False):
        """Initialize params for Class Balanced loss.
        
        Parameters
        ----------
        samples_per_cls: list or numpy.array with the number or ratio of samples
            per class of the dataset.
        loss_fn: PyTorch loss function to apply with weight params.
            The default is BCEWithLogitsLoss()
        beta: float, optional value for beta params of the loss function.
            The default is 0.9999.
        is_softmax: optional, indicate if work with softmax activation.
            The default is False.
        """
        super(ClassBalancedLoss, self).__init__()
        self.beta = float(beta)
        self.samples_per_cls = np.array(samples_per_cls)
        self.no_of_classes = self.samples_per_cls.shape[0]
        self.loss_fn = loss_fn
        self.is_softmax = is_softmax
        
        self.effective_num = 1.0 - self.beta ** np.array(samples_per_cls)
        # calculate weights per class
        weights = (1.0 - self.beta) / self.effective_num
        weights = weights / np.sum(weights) * self.no_of_classes
        self.weights = torch.tensor(weights).float()
        
        
    def forward(self, input, target): 
        """Compute the weights and loss value."""
        # if not hot_encoded
        if target.dim() < 2 and not self.is_softmax:
            target = F.one_hot(target, input.size(-1)).float()
        
        if not self.is_softmax:
            # class weight for batch
            w_idx = torch.argmax(target, dim=1)
            batch_weights = self.weights[w_idx].unsqueeze(1)
            weights = batch_weights.repeat(1, self.no_of_classes)    
            self.loss_fn.weight = weights
        else:
            self.loss_fn.weight = self.weights
            
        cb_loss = self.loss_fn(input, target)
        return cb_loss
        
class SoftmaxBCELoss(nn.BCELoss):
    def forward(self, input, target):
        pred = F.softmax(input, dim=1)
        return super(SoftmaxBCELoss, self).forward(pred, target)
        
        
if __name__ == '__main__':
    torch.manual_seed(0)
    
    alpha = 0.25
    gamma = 2.
    beta = 0.9999
    no_of_classes = 10
    samples_per_cls = [2,3,1,2,2,5,2,3,1,2]
    # samples_per_cls = np.array(sasmples_per_cls) / np.sum(samples_per_cls) #pt
    loss_type = "focal"
        
    focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
    focal_nogamma = FocalLoss(gamma=0, alpha=None)
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    focal_loss_softmax = FocalLoss(gamma=gamma, alpha=alpha, is_softmax=True)
    focal_loss_softmax_nogamma = FocalLoss(gamma=0, alpha=None, is_softmax=True)
    ce = nn.CrossEntropyLoss(reduction='mean')    
    
    bce_cb = nn.BCEWithLogitsLoss(reduction='mean')
    ce_cb = nn.CrossEntropyLoss(reduction='mean')
    
    cb_focal = ClassBalancedLoss(samples_per_cls, focal_loss, beta=beta)
    cb_focal_nobeta = ClassBalancedLoss(samples_per_cls, focal_nogamma, beta=0.)
    
    cb_sigmoid = ClassBalancedLoss(samples_per_cls, bce_cb, beta=beta)
    cb_sigmoid_nobeta = ClassBalancedLoss(samples_per_cls, bce_cb, beta=0.)
    
    cb_ce = ClassBalancedLoss(samples_per_cls, ce, beta=beta, is_softmax=True)
    cb_ce_nobeta = ClassBalancedLoss(samples_per_cls, ce, beta=0., is_softmax=True)
    
    # cb_focal_softmax = ClassBalancedLoss(samples_per_cls, focal_loss_softmax, beta=beta, is_softmax=True)
    # cb_focal_nogamma_softmax = ClassBalancedLoss(samples_per_cls, focal_loss_softmax_nogamma, beta=beta, is_softmax=True)
    # cb_focal_softmax_nobeta = ClassBalancedLoss(samples_per_cls, focal_loss_softmax, beta=0., is_softmax=True)
    # cb_focal_nogamma_softmax_nobeta = ClassBalancedLoss(samples_per_cls, focal_loss_softmax_nogamma, beta=0., is_softmax=True)
    
    
    output = torch.randn((5, no_of_classes)) # A prediction
    # output = torch.full([5, 10], 1.5)  # A prediction (logit)    
    target = torch.randint(0, no_of_classes, (5,)) # 10 classes, batch size = 5
    hot_target = F.one_hot(target, output.size(-1)).float()
    
    # calculate weights
    # with torch.no_grad():
    #     w = torch.empty_like(output).fill_(1 - alpha)
    #     w[hot_target == 1] = alpha
    # bce.weight = w
    # ce.weight = torch.tensor([1.]*no_of_classes)
    
    print('output', output)
    print('target', target)
    # print('target.dim', target.dim())
    # print('w', w, w.shape)
    print()
    print('focal_loss', focal_loss(output, target))
    print('focal_nogamma', focal_nogamma(output, target))
    print('bcross_entropy', bce(output, hot_target))
    print()
    print('focal_loss_softmax', focal_loss_softmax(output, target))
    print('focal_loss_softmax_nogamma', focal_loss_softmax_nogamma(output, target))
    print('cross_entropy_softmax', ce(output, target))
    print()
    print('cb_focal', cb_focal(output, target))
    print('cb_focal_nobeta', cb_focal_nobeta(output, target))
    print()
    print('cb_sigmoid', cb_sigmoid(output, target))
    print('cb_sigmoid_nobeta', cb_sigmoid_nobeta(output, target))
    print('bcross_entropy_cb', bce_cb(output, hot_target))
    print()
    print('cb_ce', cb_ce(output, target))
    print('cb_ce_nobeta', cb_ce_nobeta(output, target))
    print('cross_entropy_cb', ce_cb(output, target))
    # print()
    # print('cb_focal_softmax', cb_focal_softmax(output, target))
    # print('cb_focal_nogamma_softmax', cb_focal_nogamma_softmax(output, target))
    # print('cb_focal_softmax_nobeta', cb_focal_softmax_nobeta(output, target))
    # print('cb_focal_nogamma_softmax_nobeta', cb_focal_nogamma_softmax_nobeta(output, target))
 