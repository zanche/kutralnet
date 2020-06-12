#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:53:26 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from models import get_model_paths
# import matplotlib as mpl
# from cycler import cycler
# mpl.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.Dark2(np.linspace(0.1, 0.9, 10)))


class PlotHelper:
    """Helper to plot pyplot graphics storaging data in order to present later."""
    
    def __init__(self, xlabel, ylabel, title=None):
        """Initialize the helper.

        Arguments:
        ---------
            xlabel string: the value for ax.set_xlabel
            ylabel string: the value for ax.set_ylabel
            title string: the value for ax.set_title
        """
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.title = title
        self.n_items = 0
        self.ydata = []
        self.xdata = []
        self.legends = []
        self.fig = None
        self.ax = None
        
    def add_data(self, legend, xdata, ydata):
        """Add data to plot without duplicates.
        
        Arguments:
        ---------
            legend string: the value to assign as ax.plot(label=legend)
            xdata (string, number): the values for the x axis
            ydata number: the value for the y axis
        """
        duplicated = False
        if not legend in self.legends:
            self.legends.append(legend)
            self.n_items += 1
        else:
            idx = self.legends.index(legend)
            duplicated = True
            
        if not xdata in self.xdata:
            self.xdata.append(xdata)            
            
        if duplicated:
            if not isinstance(self.ydata[idx], list):
                self.ydata[idx] = [self.ydata[idx]]
            self.ydata[idx].append(ydata)
        else:
            self.ydata.append(ydata)
            
    def make_title(self, ax, fontsize=18, **kwargs):        
        """Add title to the figure.
        
        Arguments:
        ---------
            ax Axes: the pyplot axes instance.
            fontsize int: the fontsize for title
        """
        if not self.title is None:
            ax.set_title(self.title, fontsize=fontsize, **kwargs)
        
    def plot_bar(self, width=0.25, ylim=None, offset=0.02):
        """
        Plot a Bar graph.

        Parameters
        ----------
            width float:(optional) the bar's width. The default is 0.25.
            ylim list: (optional) specify the y axis limits. The default is None.
            offset float: (optional) the distance between bars. The default is 0.02.

        Returns
        -------
            fig matplotlib.Figure: the figure to be reused
            ax Axes: the axes to be reused
        """
        xis_string = isinstance(self.xdata[0], str)
        fig, ax = plt.subplots()
        if xis_string:
            # the label locations
            bar_labels = self.xdata
            x = np.arange(len(bar_labels), dtype=float)
            x_val = x - width / 2 * (self.n_items -1) # distance from center
            x_val -= offset / 2 * (self.n_items -1) # distance between bars        
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels)  
        else:
            x_val = self.xdata
        
        for i in range(self.n_items):
            ax.bar(x_val, self.ydata[i], width, label=self.legends[i], zorder=2)
            if xis_string:
                x_val += width + offset
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(self.ylabel, fontsize=14)
        ax.set_xlabel(self.xlabel, fontsize=14)
        if not ylim is None:
            ax.set_ylim(ylim)
        self.make_title(ax)
        ax.legend(loc="lower right", prop={'size': 14})
        ax.grid()        
        fig.tight_layout()
        self.fig, self.ax = fig, ax
        return fig, ax
    
    def plot_bar_percentage(self, width=0.25, ylim=None, offset=0.02):
        """Plot a Bar graph with percentage representation of axis."""
        fig, ax = self.plot_bar(width=width, ylim=ylim, offset=offset)        
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        return fig, ax
        
    def plot_roc(self, lw=2):
        """Plot the ROC curve."""        
        fig, ax = plt.subplots()
        for i in range(self.n_items):
            fpr = self.xdata[i]
            tpr = self.ydata[i]
            legend = self.legends[i]
            
            ax.plot(fpr, tpr, lw=lw, label=legend, linestyle='-.')
        
        # baseline
        ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
        ax.set_ylim([ 0.0, 1.01])
        ax.set_xlim([-0.01, 1.0])
        ax.grid()
        ax.set_xlabel(self.xlabel, fontsize=14)
        ax.set_ylabel(self.ylabel, fontsize=14)
        
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend(loc="lower right", prop={'size': 14})
        fig.tight_layout()
        self.fig, self.ax = fig, ax
        return fig, ax
    
    def save(self, file_path):
        """Save the current Figure into file_path."""
        self.fig.savefig(file_path)

def get_results_path(models_root, create_path=False):
    """Get (and create) results path for graphs storage."""
    # main folder
    models_result_path = os.path.join(models_root, 'results')
    # if must create
    if create_path:
        # models root
        if not os.path.exists(models_root):
            os.makedirs(models_root)
        # model reulst path folder
        if not os.path.exists(models_result_path):
            os.makedirs(models_result_path)
    
    else:
        # dir exists?
        if not os.path.isdir(models_result_path):
            models_result_path = None
            
    return models_result_path

def get_data(models_root, model_name, dataset_name, version=None):
    """Get the training and testing data from a model."""
    save_path, _ = get_model_paths(models_root, model_name, 
                                                 dataset_name, version)
    # check if was trained
    if save_path is None:
        return None, None, None
    # training summary
    training_data = pd.read_csv(os.path.join(save_path, 'training_summary.csv'),
                                header=None)
    # testing summary
    testing_data = pd.read_csv(os.path.join(save_path, 'testing_summary.csv'),
                               header=None)
    # ROC values
    with open(os.path.join(save_path, 'roc_summary.pkl'), 'rb') as f:
        roc_data = pickle.load(f)
    
    return training_data, testing_data, roc_data

def get_names(training_data):
    """Get the model and dataset names from summary."""
    dataset_name = training_data.loc[3, 1]
    model_name = training_data.loc[1, 1]
    return model_name, dataset_name

def get_validation_acc(training_data):
    """Get the validation data from summary."""
    val_acc = training_data.loc[15, 1]
    best_epoch = training_data.loc[16, 1]
    return float(val_acc), int(best_epoch)
    
def get_testing_acc(testing_data):
    """Get the testing data from summary."""
    test_acc = testing_data.iloc[3, 1]
    auroc_val = testing_data.iloc[2, 1]
    return float(test_acc), float(auroc_val)

def plot_all(models_root, datasets, models, version=None, name_prefix=None):
    """Plot all the required graphs once (and save it).
    
    If name_prefix is set, the graphics will be saved inside 
    {models_root}/results folder with {name_prefix}_(test|validation).pdf
    """
    must_save = not name_prefix is None
    results_path = get_results_path(models_root, create_path=must_save)
    
    plotter_val = PlotHelper('Dataset', 'Validation Accuracy')
    plotter_test = PlotHelper('Dataset', 'Test Accuracy')
    
    for dataset_id in datasets:
        plotter_roc = PlotHelper('False Positive Rate', 'True Positive Rate')
        
        for model_id in models:
            training_data, testing_data, roc_data = get_data(models_root, 
                                                model_id, dataset_id, version)
            model_name, dataset_name = get_names(training_data)
            val_acc, best_epoch = get_validation_acc(training_data)
            test_acc, auroc_val = get_testing_acc(testing_data)
            
            plotter_val.add_data(model_name, dataset_name, val_acc)
            plotter_test.add_data(model_name, dataset_name, test_acc)
            
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            roc_auc = roc_data['roc_auc']
            legend = '{} AUROC={:.2f}'.format(model_name, roc_auc)
            plotter_roc.add_data(legend, fpr, tpr)
            
        plotter_roc.title = "ROC curve for {} dataset".format(dataset_name)
        plotter_roc.plot_roc()
        if must_save:
            file_name = "{}_{}_ROCcurve.pdf".format(name_prefix, dataset_name)
            plotter_roc.save(os.path.join(results_path, file_name))
        
    plotter_val.plot_bar_percentage(ylim=[0.3, 1.01])
    plotter_test.plot_bar_percentage(ylim=[0.3, 1.01])
    
    if must_save:
        file_name = "{}_validation_accuracy.pdf".format(name_prefix)
        plotter_val.save(os.path.join(results_path, file_name))
        
        file_name = "{}_test_accuracy.pdf".format(name_prefix,)
        plotter_test.save(os.path.join(results_path, file_name))
    
def plot_history(history, folder_path=None):
    """Plot history file from training."""
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'accuracy.png'))

    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'loss.png'))

    plt.show()
# end plot_history

def plot_samples(data):
    """Plot a sample of image from data."""
    fig = plt.figure()

    for i in range(len(data)):
        sample = data[i]

        print(i, sample[0].shape, sample[1].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        label = 'Fire' if sample[1] == 1 else 'Nofire'
        ax.set_title('Sample {}'.format(label))
        ax.axis('off')
        img = sample[0].transpose(2, 0)
        plt.imshow(img.transpose(0, 1))

        if i == 3:
            plt.show()
            break
# end show_samples

if __name__ == '__main__':
    root_path = os.path.join('..')
    models_root = os.path.join(root_path, 'models')
    print('Root path:', root_path)
    print('Models path:', models_root)
    version = None
    
    # baseline comparison
    models = ['firenet_tf', 'kutralnet', 'octfiresnet', 'resnet']
    datasets = ['fismo']# ['firenet', 'fismo', 'fismo_black']
    plot_all(models_root, datasets, models, version=version, 
             name_prefix='baseline')
    