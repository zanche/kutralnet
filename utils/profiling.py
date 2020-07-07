import os
import torch
import pandas as pd
from thop import profile, clever_format
from models import get_model
from models import get_model_params

def model_profile(model, config):
    img_dims = config['img_dims']
    x = torch.randn(1, 3, img_dims[0], img_dims[1])
    flops, params = profile(model, verbose=False,
            inputs=(x, ),
            custom_ops={torch.nn.Dropout2d: None})
    flops, params = clever_format([flops, params], "%.4f")
    print('{}_{}: {} flops, {} params'.format(config['model_name'],
                                             img_dims, flops, params))
    return flops, params

def load_model_profile(model_name, num_classes=2, extra_params=None):
    # model selection
    config = get_model_params(model_name)
    model = get_model(model_name, extra_params=extra_params)
    return model_profile(model, config)

def model_size(model_path):
    # size
    file_size = os.stat(model_path).st_size
    file_size /= 1024 **2
    fs_string = '{:.2f}MB'.format(file_size)
    print(fs_string)
    return fs_string

def model_performance(history_file):
    # gets model info and validation accuracy
    if isinstance(history_file, str):
        history = pd.read_csv(history_file)
    elif isinstance(history_file, dict):
        history = pd.DataFrame(data=history_file)
    else:
        raise ValueError('The history must be either the csv file path or a dict.')
        
    #history.sort_values('val_acc'), inplace=True)
    best_row = history.sort_values('val_acc').tail(1)
    best_val_acc, best_ep = best_row.iloc[0]['val_acc'], best_row.index[0]
    best_ep = int(best_ep) +1
    print('Max validation accuracy {:.4f} reached on epoch {}'.format(
                    best_val_acc, best_ep))
    return best_val_acc, best_ep
    