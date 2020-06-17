import os
import argparse
from utils.plotting import plot_all


parser = argparse.ArgumentParser(description='Summary plot script')
parser.add_argument('--models', default=['baseline', 'portable'], nargs='+',
                    help='the trained models ID presented, i.e, --models \
                    resnet kutralnet kutralnet_octave ocfiresnet')
parser.add_argument('--datasets', default=['fismo'], nargs='+',
                    help='the datasets ID used for training as --datasets fismo (firenet (fismo_black))')
parser.add_argument('--versions', nargs='*',
                    help='the trained version be presented as --versions v1 v2 v3')
parser.add_argument('--graphs', default='val test roc', nargs='+',
                    help='the graphs to be presented, i.e., --graphs val testr roc')
parser.add_argument('--models-path', default='models',
                    help='the path where are stored the models')
parser.add_argument('--save-prefix', default='test',
                    help='the prefix to save the figures')
args = parser.parse_args()

# user's selections
models = args.models #'kutralnet'
datasets = args.datasets #'fismo'
versions = args.versions #None    
graphs = args.graphs
save_prefix = args.save_prefix
models_root = args.models_path
print('Models path:', models_root)

if versions is None or len(versions) == 0:
     versions = [None]

print('models', models)
print('datasets', datasets)
print('versions', versions)
print('graphs', graphs)

if 'baseline' in models:
    m = ['firenet_tf', 'kutralnet', 'octfiresnet', 'resnet']
    d = ['firenet', 'fismo', 'fismo_black']
    models.remove('baseline')
    plot_all(models_root, d, m, graphs=graphs, versions=versions,
         name_prefix='baseline', title=False)
    
if 'portable' in models:
    m = ['kutralnet', 'kutralnet_mobile', 'kutralnetoct', 'kutralnet_mobileoct']
    d = ['fismo', 'fismo_balanced', 'fismo_balanced_black']
    models.remove('portable')
    plot_all(models_root, d, m, graphs=graphs, versions=versions,
         name_prefix='portable', title=False)

if len(models) > 0:
    plot_all(models_root, datasets, models, graphs=graphs, versions=versions,
             name_prefix=save_prefix, title=False)
