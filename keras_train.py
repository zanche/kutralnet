import os
import time
import pickle
import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from datasets import datasets
from models import get_model_paths
from models.firenet_tf import firenet_tf
from utils.dataset import load_dataset
from utils.dataset import load_firenet_test_dataset
from utils.dataset import preprocess
from keras_profile import model_size
from utils.training import add_bool_arg
from utils.training import save_csv
from utils.plotting import plot_history


parser = argparse.ArgumentParser(description='Classification models training script, TF versions')
parser.add_argument('--base-model', metavar='BM', default='firenet_tf',
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

# Set a seed value
seed_value= 666
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)
# 5. For layers that introduce randomness like dropout, make sure to set seed values
# model.add(Dropout(0.25, seed=seed_value))
#6 Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


must_train = True
must_test = True
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
use_cuda = 'Unknown'

# dataset selection
dataset_name = datasets[dataset_id]['name']
num_classes = datasets[dataset_id]['num_classes']
ds_folder, get_dataset = load_dataset(dataset_id)

# save models direclty in the repository's folder
root_path = os.path.join('.')
models_root = os.path.join(root_path, 'models')
print('Root path:', root_path)
print('Models path:', models_root)
save_path, _ = get_model_paths(models_root, model_id, dataset_id, 
                               version=version, create_path=True)
model_path = os.path.join(save_path, 'model_{}.h5'.format(model_id))

### Training
if must_train:

    ds_path = os.path.join('.', 'datasets', ds_folder)
    x_train, y_train, x_val, y_val = get_dataset(ds_path, resize=(64,64))

    # Normalize data.
    x_train = preprocess(x_train)
    x_val = preprocess(x_val)

    # summary
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(y_train[y_train==1].shape[0], 'fire')
    print(y_train[y_train==0].shape[0], 'no_fire')

    print('x_val shape:', x_val.shape)
    print(x_val.shape[0], 'test samples')
    print(y_val[y_val==1].shape[0], 'fire')
    print(y_val[y_val==0].shape[0], 'no_fire')

    num_classes = 2
    input_shape = x_train.shape[1:]
    print('num_classes', num_classes, 'input_shape', input_shape)

    # Convert class vectors to binary class matrices.
    # y_train = utils.to_categorical(y_train, num_classes)
    # y_val = utils.to_categorical(y_val, num_classes)

    def prepare_callbacks(save_dir, suffix):
        # Prepare model model saving directory.
        model_name = 'model_%s.h5' % suffix
        history_name = 'history.csv'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, model_name)
        historypath = os.path.join(save_dir, history_name)

        # Prepare callbacks for saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)

        csv_logger = CSVLogger(filename=historypath,
                                separator=',',
                                append=False)

        return [csv_logger, checkpoint]
    # end prepare_callbacks

    model = firenet_tf(input_shape=input_shape)
    criterion = 'sparse_categorical_crossentropy'
    optimizer = 'adam'
    model.compile(loss=criterion,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    #
    print('Initiating training, models will be saved at {}'.format(save_path))
    time_elapsed = 0
    since = time.time()
    # since = time.time()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=(x_val, y_val), callbacks=prepare_callbacks(save_path, model_id))

    best_ep = np.argmax(history.history['val_acc'])
    best_acc = history.history['val_acc'][best_ep]

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy on epoch {}: {:4f}'.format(best_ep, best_acc))
    
    # model summary
    print("Model's on-disk size", end=' ')
    model_disk_size = model_size(model_path)
    
    # training summary save
    keys = ['Model ID', 'Model name', 'Training dataset ID', 'Training dataset name', 
            'Version', 'Using CUDA', 'Epochs', 'Batch size', 'Shuffle dataset',
            'Loss function', 'Optimizer', 'Scheduler', 'Model parameters', 
            'Model FLOPS', 'Training time (s)', 'Validation accuracy', 'Best ep',
            'Model on-disk size']
    values = [model_id, 'FireNet', dataset_id, dataset_name, 
              str(version), use_cuda, epochs, batch_size, shuffle_dataset,
              criterion, optimizer, str(None), '646.82K', #taken from model.summary()
              'Undefined', time_elapsed, best_acc, best_ep, 
              model_disk_size]
    
    training_summary = list(zip(keys, values))
    save_csv(training_summary, 
             file_path=os.path.join(save_path, 'training_summary.csv'),
             header=False,
             index=False)

    plot_history(history.history, folder_path=save_path)


### Test
if must_test:

    firenet_path = os.path.join('.', 'datasets', 'FireNetDataset')
    x_test, y_true = load_firenet_test_dataset(firenet_path, resize=(64,64))
    # dataset selection
    test_dataset_id = 'firenet_test'
    test_dataset = datasets[test_dataset_id]['class']
    test_num_classes = datasets[test_dataset_id]['num_classes']


    # Normalize data.
    x_test = preprocess(x_test)

    # summary
    print('x_test shape:', x_test.shape)
    print(x_test.shape[0], 'test samples')
    print(y_true[y_true==1].shape[0], 'fire')
    print(y_true[y_true==0].shape[0], 'no_fire')

    num_classes = 2
    input_shape = x_test.shape[1:]
    print('num_classes', num_classes, 'input_shape', input_shape)

    model = firenet_tf(input_shape=input_shape)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.load_weights(model_path)

    #Confusion Matrix and Classification Report
    since = time.time()
    y_pred = model.predict(x_test, verbose=1)
    test_time = time.time() - since            
    
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
    target_names = ['No Fire', 'Fire']
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
    test_accuracy = test_results['accuracy']
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
