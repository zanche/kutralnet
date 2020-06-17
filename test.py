import os
import json
import torch
import pickle
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from datasets import datasets
from models import get_model
from models import get_model_paths
from utils.training import add_bool_arg
from utils.training import test_model
from utils.training import save_csv


parser = argparse.ArgumentParser(description='Fire classification test')
parser.add_argument('--model', metavar='MODEL_ID', default='kutralnet',
                    help='the model ID for training')
parser.add_argument('--dataset', metavar='DATASET_ID', default='fismo',
                    help='the dataset ID for training')
parser.add_argument('--version', metavar='VERSION_ID', default=None,
                    help='the training version')
parser.add_argument('--batch-size', default=32, type=int,
                    help='the number of items in the batch')
parser.add_argument('--models-path', default='models',
                    help='the path where storage the models')
parser.add_argument('--model-params', default=None,
                    help='the params to instantiate the model')
add_bool_arg(parser, 'preload-data', default=True, help='choose if load or not the dataset on-memory')
add_bool_arg(parser, 'seed', default=True, help='choose if set or not a seed for random values')
args = parser.parse_args()

# user's selection
model_id = args.model #'kutralnet'
train_dataset_id = args.dataset #'fismo'
version = args.version #None
models_root = args.models_path
preload_data = bool(args.preload_data) #True # load dataset on-memory
batch_size = args.batch_size #32
must_seed = bool(args.seed)
extra_params = args.model_params

if not extra_params is None:
    extra_params = json.loads(extra_params)
# cuda if available
use_cuda = torch.cuda.is_available()
torch_device = 'cpu'

if use_cuda:
    torch_device = 'cuda'

if must_seed:
    # Seed
    seed_val = 666
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False        

# dataset selection
test_dataset_id = 'firenet_test'
test_dataset = datasets[test_dataset_id]['class']
test_num_classes = datasets[test_dataset_id]['num_classes']

# model load
model, config = get_model(model_id, num_classes=test_num_classes, 
                          extra_params=extra_params)

# dataset load
transform_compose = config['preprocess_test']
dataset = test_dataset(transform=transform_compose, preload=preload_data)

# read models direclty from the repository's folder
print('Models path:', models_root)
save_path, model_path = get_model_paths(models_root, model_id, 
                                                 train_dataset_id, version)
# load trained model
print('Loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path, 
                            map_location=torch.device(torch_device)))
# testing
y_true, y_pred, test_accuracy, test_time = test_model(model, dataset, 
                                           batch_size=batch_size, 
                                           use_cuda=use_cuda)

# just percentage to fire class
y_score = [y[1] for y in y_pred]
# Compute ROC curve and ROC area:
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
roc_summary = {
    'fpr': fpr,
    'tpr': tpr,
    'roc_auc': roc_auc
}

# save roc data
with open(os.path.join(save_path, 'roc_summary.pkl'), 'wb') as f:
    pickle.dump(roc_summary, f, pickle.HIGHEST_PROTOCOL)
    
print('Area under the ROC curve', roc_auc)

# confusion matrix
print('Classification Report')
target_names = [ dataset.labels[label]['name'] for label in dataset.labels.keys() ]
# class discretize
y_pred_class = np.argmax(y_pred, axis=1)
# printing purpose only
class_report = classification_report(y_true, y_pred_class,
                        target_names=target_names)#, output_dict=True)

print(class_report)
test_results = classification_report(y_true, y_pred_class,
                        target_names=target_names, output_dict=True)

# testing summary
keys = ['Version', 'Testing time (s)', 'AUROC value', 'Testing accuracy',
        'Using CUDA', 'Batch size']
test_accuracy /= 100
values = [str(version), test_time, roc_auc, test_accuracy,
          use_cuda, batch_size]

for label in target_names:
    for k in test_results[label]:
        keys.append("{} {}".format(label, k))
        values.append(test_results[label][k])

testing_summary = list(zip(keys, values))

save_csv(testing_summary, 
         file_path=os.path.join(save_path, 'testing_summary.csv'),
         header=False,
         index=False)
