import os
from utils.plotting import plot_all

root_path = os.path.join('.')
models_root = os.path.join(root_path, 'models')
print('Root path:', root_path)
print('Models path:', models_root)
version = None

# baseline comparison
models = ['firenet_tf', 'kutralnet', 'octfiresnet', 'resnet']
datasets = ['firenet', 'fismo', 'fismo_black']
plot_all(models_root, datasets, models, version=version, 
         name_prefix='baseline', title=False)

# portable comparison
models = ['kutralnet', 'kutralnet_mobile', 'kutralnetoct', 'kutralnet_mobileoct']
datasets = ['fismo', 'fismo_balanced', 'fismo_balanced_black']
plot_all(models_root, datasets, models, version=version, 
         name_prefix='portable', title=False)
