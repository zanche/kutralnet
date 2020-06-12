import os
import torch
import argparse
import numpy as np
from datasets import datasets
from models import get_model
from models import get_model_paths
from utils.training import add_bool_arg
from utils.training import train_model
from utils.training import save_csv
from utils.training import BestModelSaveCallback
from utils.profiling import model_size
from utils.profiling import model_performance
from utils.profiling import load_model_profile
from utils.plotting import plot_history


parser = argparse.ArgumentParser(description='Classification models training script')
parser.add_argument('--base-model', metavar='BM', default='kutralnet',
                    help='the model ID for training')
parser.add_argument('--epochs', metavar='EP', default=100, type=int,
                    help='the number of maximum iterations')
parser.add_argument('--batch-size', metavar='BS', default=32, type=int,
                    help='the number of items in the batch')
parser.add_argument('--dataset', metavar='DS', default='fismo',
                    help='the dataset ID for training')
parser.add_argument('--version', metavar='VER', default=None,
                    help='the training version')
add_bool_arg(parser, 'preload-data', default=False, help='choose if load or not the dataset on-memory')
add_bool_arg(parser, 'pin-memory', default=False, help='choose if pin or not the data into CUDA memory')
args = parser.parse_args()

# Seed
seed_val = 666
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed_val)
np.random.seed(seed_val)

if use_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# user's selections
model_id = args.base_model #'kutralnet'
dataset_id = args.dataset #'fismo'
version = args.version #None
# train config
epochs = args.epochs #100
batch_size = args.batch_size #32
shuffle_dataset = True
preload_data = bool(args.preload_data) #False # load dataset on-memory
pin_memory = bool(args.pin_memory) #False # pin dataset on-memory

# dataset selection
dataset_name = datasets[dataset_id]['name']
base_dataset = datasets[dataset_id]['class']
num_classes = datasets[dataset_id]['num_classes']

# model load
model, config = get_model(model_id, num_classes=num_classes)

# common preprocess
transform_train = config['preprocess_train']
transform_val = config['preprocess_val']

# dataset read
train_data = base_dataset(transform=transform_train, preload=preload_data)
val_data = base_dataset(purpose='val', transform=transform_val, preload=preload_data)

# loss function
criterion = config['criterion']
opt_args = {'params': model.parameters()}
opt_args.update(config['optimizer_params'])

# optimizer
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
root_path = os.path.join('.')
models_root = os.path.join(root_path, 'models')
print('Root path:', root_path)
print('Models path:', models_root)
save_path, model_path = get_model_paths(models_root, model_id, 
                                                 dataset_id, version=version,
                                                 create_path=True)
save_callback = BestModelSaveCallback(model_path)

# training
model_flops, model_params = load_model_profile(model_id, num_classes=num_classes)
print('Initiating training, models will be saved at {}'.format(save_path))
history, best_model, time_elapsed = train_model(model, criterion, optimizer, train_data, val_data,
            epochs=epochs, batch_size=batch_size, shuffle_dataset=shuffle_dataset, scheduler=scheduler,
            use_cuda=use_cuda, pin_memory=pin_memory, callbacks=[save_callback])

# model summary
print("Model's on-disk size", end=' ')
model_disk_size = model_size(model_path)

# history save
save_csv(history, file_path=os.path.join(save_path, 'history.csv'))
best_acc, best_ep = model_performance(history)

# training summary save
keys = ['Model ID', 'Model name', 'Training dataset ID', 'Training dataset name', 
        'Version', 'Using CUDA', 'Epochs', 'Batch size', 'Shuffle dataset',
        'Loss function', 'Optimizer', 'Scheduler', 'Model parameters', 
        'Model FLOPS', 'Training time (s)', 'Validation accuracy', 'Best ep',
        'Model on-disk size']
values = [model_id, config['model_name'], dataset_id, dataset_name, 
          str(version), use_cuda, epochs, batch_size, shuffle_dataset,
          criterion, optimizer, str(scheduler_info), model_params, 
          model_flops, time_elapsed, best_acc, best_ep, 
          model_disk_size]

training_summary = list(zip(keys, values))
save_csv(training_summary, 
         file_path=os.path.join(save_path, 'training_summary.csv'),
         header=False,
         index=False)

plot_history(history, folder_path=save_path)
