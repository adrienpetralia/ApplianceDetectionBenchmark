import torch
import torch.nn as nn

from Utils.Layers.ConvLayers import Conv1dSame

class ConvNet(nn.Module):
    def __init__(self, in_channels=1, nb_class=2):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            Conv1dSame(in_channels=in_channels, out_channels=128, kernel_size=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )
            
        self.layer2 = nn.Sequential(
            Conv1dSame(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )

        self.layer3 = nn.Sequential(
            Conv1dSame(in_channels=256, out_channels=128, kernel_size=3),
            nn.ReLU(),
            )
            
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(128, nb_class)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.GAP(x)
        x = torch.flatten(x, 1)
        
        return self.linear(x)