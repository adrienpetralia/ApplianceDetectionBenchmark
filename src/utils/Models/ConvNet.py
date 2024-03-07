import torch
import torch.nn as nn

class Conv1dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True,
                 padding_layer=torch.nn.ReflectionPad1d):
        super(Conv1dSame, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        
        if dilation > 1:
            ka = ka * dilation
            kb = kb * dilation
        
        self.net = torch.nn.Sequential(padding_layer((ka,kb)),
                                       torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                                                       dilation=dilation, bias=bias)
                                       )

    def forward(self, x):
        return self.net(x)
<<<<<<< HEAD

=======
>>>>>>> dd81df6aded289a7cd9dc043d4220cc4d8001285

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
