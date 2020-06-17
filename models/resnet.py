from torch import nn
from torchvision.models import resnet50
from torchvision.models import resnet18

def resnet_sharma(classes):
    """
    Transfer learning from ResNet50 modified version to fire classification used in
    Deep Convolutional Neural Networks for Fire Detection in Images (2017)
    """
    resnet = resnet50(pretrained=True)
    num_ftrs = resnet.fc.in_features
    # modify the last layer to add another FC
    resnet.fc = nn.Linear(num_ftrs, 4096)
    # freeze for transfer learning
    freeze_layer = 9

    for i, child in enumerate(resnet.children()):
        if i < freeze_layer:
            for param in child.parameters():
                param.requires_grad = False

    return nn.Sequential(
        resnet,
        nn.Linear(4096, classes)
    )
# end resnet_sharma

def ResNet18(classes, freeze_layer=9):
    """
    ResNet18 for use with Transfer-learning
    """
    resnet = resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    # modify the last layer to add another FC
    resnet.fc = nn.Linear(num_ftrs, 1024)   

    # freeze layers
    for i, child in enumerate(resnet.children()):
        if i < freeze_layer:
            for param in child.parameters():
                param.requires_grad = False

    return nn.Sequential(
        resnet,
        nn.Linear(1024, classes)
    )

# end ResNet18
