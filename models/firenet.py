import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, features_in, features_out):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(features_in, features_out, kernel_size=3)
        self.avg_pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(p=.5)
    # end __init__

    def forward(self, x):
        z = F.relu(self.conv1(x))
        z = self.avg_pool(z)
        y = self.dropout(z)
        return y
    # end forward
# end ConvBlock

class FireNet(nn.Module):
    def __init__(self, classes):
        super(FireNet, self).__init__()
        self.block1 = ConvBlock(3, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)
        self.linear1 = nn.Linear(64 * 6 * 6, 256)
        self.dropout = nn.Dropout(p=.2)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, classes)
        self._init_params()
    # end __init__

    def forward(self, x):
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = z.flatten(start_dim=1)
        z = F.relu(self.linear1(z))
        z = self.dropout(z)
        z = F.relu(self.linear2(z))
        y = self.linear3(z)
        return y
    # end forward

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    # end _init_params
# end FireNet

