import os
import ast
import torch
import argparse
import numpy as np
from datasets import get_dataset
from datasets import get_preprocessing
from models import get_model
from models import get_model_paths
from models import get_model_params
from models import get_loss
from models import get_activation
from utils.training import add_bool_arg
from utils.training import train_model
from utils.training import save_csv
from utils.training import BestModelSaveCallback
from utils.profiling import model_size
from utils.profiling import model_performance
from utils.profiling import load_model_profile
from utils.plotting import plot_history


parser = argparse.ArgumentParser(description='Classification models training script')
parser.add_argument('--model', metavar='MODEL_ID', default='kutralnet',
                    help='the model ID for training')
parser.add_argument('--model-params', default=None, nargs='*',
                    help='the params to instantiate the model as KEY=VAL')
parser.add_argument('--activation', default='ce_softmax',
                    help='the activation and loss function for the model as LOSS_ACTIVATION pattern')
parser.add_argument('--loss-params', default=None, nargs='*',
                    help='the params to instantiate the cost function as KEY=VAL')
parser.add_argument('--dataset', metavar='DATASET_ID', default='fismo',
                    help='the dataset ID for training')
parser.add_argument('--dataset-flags', default=None, nargs='*',
                    help='the datasets flags to instaciate the dataset, this \
                        flags can be: \
                            - (no_)one_hot: to one-hot encode or not the labels.\
                            - (no_)distributed: to use or not a distributed representation.\
                            - (no_)multi_label: to allow or not the use of multi-label images.')
parser.add_argument('--epochs', default=100, type=int,
                    help='the number of maximum iterations')
parser.add_argument('--batch-size', default=32, type=int,
                    help='the number of items in the batch')
parser.add_argument('--version', metavar='VERSION_ID', default=None,
                    help='the training version')
parser.add_argument('--models-path', default='models',
                    help='the path where storage the models')
add_bool_arg(parser, 'preload-data', default=False, help='choose if load or not the dataset on-memory')
add_bool_arg(parser, 'pin-memory', default=False, help='choose if pin or not the data into CUDA memory')
add_bool_arg(parser, 'seed', default=True, help='choose if set or not a seed for random values')
args = parser.parse_args()

# user's selections
model_id = args.model #'kutralnet'
dataset_id = args.dataset #'fismo'
version = args.version #None
models_root = args.models_path
# train config
activation_fn = args.activation # 'ce_softmax'
epochs = args.epochs #100
batch_size = args.batch_size #32
shuffle_dataset = True
preload_data = bool(args.preload_data) #False # load dataset on-memory
pin_memory = bool(args.pin_memory) #False # pin dataset on-memory
must_seed = bool(args.seed) #True # set seed value
model_params = args.model_params
dataset_flags = args.dataset_flags
loss_params = args.loss_params
    
# cuda if available
use_cuda = torch.cuda.is_available()

# Seed
if must_seed:
    seed_val = 666
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# models parameters
config = get_model_params(model_id)

# common preprocess
transform_params = dict(img_dims=config['img_dims'])
transform_train = get_preprocessing(config['preprocess_train'], transform_params)
transform_val = get_preprocessing(config['preprocess_val'], transform_params)

# dataset read
data_params = dict(transform=transform_train, preload=preload_data)
data_params.update(dict(dataset_flags=dataset_flags))
train_data = get_dataset(dataset_id, params=data_params)

data_params.update(dict(purpose='val'))
val_data = get_dataset(dataset_id, data_params)

num_classes = train_data.num_classes

# model load
model = get_model(model_id, num_classes=num_classes, extra_params=model_params)

# cost function
loss_extra_params = dict()
if not loss_params is None:
    for _p in loss_params:
        key, val = _p.split("=")
        try:
            val = ast.literal_eval(val)                
        except:
            pass
        loss_extra_params[key] = val

# class balanced variation
if 'cb_' in activation_fn:
    print('Class balanced variation')
    samples_class = train_data.samples_by_class
    ref_samples = []
    
    for k, sample in samples_class.items():
        ref_samples.append(sample['p'])
        
    loss_extra_params['samples_per_cls'] = ref_samples
    loss_extra_params['distributed_rep'] = train_data.distributed
    
criterion = get_loss(activation_fn, loss_extra_params)
print("Using", criterion)
activation = get_activation(activation_fn)

# optimizer
opt_args = {'params': model.parameters()}
opt_args.update(config['optimizer_params'])
optimizer = config['optimizer'](**opt_args)

# scheduler
scheduler = None
scheduler_info = None
if config['scheduler'] is not None:
    sched_args = {'optimizer': optimizer}
    sched_args.update(config['scheduler_params'])
    scheduler = config['scheduler'](**sched_args)
    # scheduler training summary
    scheduler_info = scheduler.__class__.__name__
    scheduler_info += str(config['scheduler_params'])
    scheduler_info = scheduler_info.replace("{", "(").replace("}", ")")
    scheduler_info = scheduler_info.replace("'", "").replace(": ", "=")

# save models direclty in the repository's folder
print('Models path:', models_root)
save_path, model_path = get_model_paths(models_root, model_id, 
                                                 dataset_id, version=version,
                                                 create_path=True)
save_callback = BestModelSaveCallback(model_path)

# training
model_flops, model_params = load_model_profile(model_id, num_classes=num_classes)
print('Initiating training, models will be saved at {}'.format(save_path))
train_summ = train_model(model, 
                            criterion, 
                            optimizer, 
                            activation, 
                            train_data, 
                            val_data,
                            epochs=epochs, 
                            batch_size=batch_size, 
                            shuffle_dataset=shuffle_dataset, 
                            scheduler=scheduler,
                            use_cuda=use_cuda, 
                            pin_memory=pin_memory, 
                            callbacks=[save_callback])

history, best_model, time_elapsed = train_summ
# model summary
print("Model's on-disk size", end=' ')
model_disk_size = model_size(model_path)

# history save
save_csv(history, file_path=os.path.join(save_path, 'history.csv'))
best_acc, best_ep = model_performance(history)

# training summary save
keys = ['Model ID', 'Model name', 'Training dataset ID', 'Training dataset', 
        'Version', 'Using CUDA', 'Epochs', 'Batch size', 'Shuffle dataset',
        'Loss function', 'Activation', 'Optimizer', 'Scheduler', 'Model parameters', 
        'Model FLOPS', 'Training time (s)', 'Validation accuracy', 'Best ep',
        'Model on-disk size']
values = [model_id, config['model_name'], dataset_id, train_data, 
          str(version), use_cuda, epochs, batch_size, shuffle_dataset,
          criterion, activation, optimizer, str(scheduler_info), model_params, 
          model_flops, time_elapsed, best_acc, best_ep, 
          model_disk_size]

training_summary = list(zip(keys, values))
save_csv(training_summary, 
         file_path=os.path.join(save_path, 'training_summary.csv'),
         header=False,
         index=False)

plot_history(history, folder_path=save_path)
