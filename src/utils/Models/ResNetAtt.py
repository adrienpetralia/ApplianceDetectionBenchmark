import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)

class ResUnit(nn.Module):       
    def __init__(self, c_in, c_out, k=8, dilation=1, stride=1, bias=True, dp_rate=0.1):
        super(ResUnit,self).__init__()
        
        self.layers = nn.Sequential(Conv1dSamePadding(in_channels=c_in, out_channels=c_out,
                                                      kernel_size=k, dilation=dilation, stride=stride, bias=bias),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(c_out),
                                    nn.Dropout(dp_rate)
                                    )
        if c_in > 1 and c_in!=c_out:
            self.match_residual=True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual=False
            
    def forward(self,x):
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)
            
            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))

class DilatedBlock(nn.Module):  
    def __init__(self, c_in=24, c_out=24, 
                 kernel_size=8, dilation_list=[1, 2, 4, 8], dp_rate=0.1):
        super(DilatedBlock,self).__init__()
 
        layers = []
        for i, dilation in enumerate(dilation_list):
            if i==0:
                layers.append(ResUnit(c_in, c_out, k=kernel_size, dilation=dilation, dp_rate=dp_rate))
            else:
                layers.append(ResUnit(c_out, c_out, k=kernel_size, dilation=dilation, dp_rate=dp_rate))
        self.network = torch.nn.Sequential(*layers)
            
    def forward(self,x):
        x = self.network(x)
        return x
    
class AttentionBlock(nn.Module):  
    def __init__(self, c_in=24, c_out=24, kernel_size=8, dp_rate=0.1):
        super(AttentionBlock,self).__init__()
        
        self.ResUnit1 = ResUnit(c_in, c_out, k=kernel_size)
        self.ResUnit2 = ResUnit(c_in, c_out, k=kernel_size)
        self.ResUnit3 = ResUnit(c_in, c_out, k=kernel_size)
        self.ResUnit4 = ResUnit(c_in, c_out, k=kernel_size)
        
        self.max_pool1 = nn.MaxPool1d(kernel_size, return_indices=True)
        self.max_pool2 = nn.MaxPool1d(kernel_size, return_indices=True)
        
        self.max_unpool1 = nn.MaxUnpool1d(kernel_size)
        self.max_unpool2 = nn.MaxUnpool1d(kernel_size)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_input):
        x = self.ResUnit1(x_input)
        S1 = x.size()
        new_x, indices1 = self.max_pool1(x)
        x = self.ResUnit2(new_x)
        S2 = x.size()
        x, indices2 = self.max_pool2(x)
        x = self.ResUnit3(x)
        x = self.max_unpool1(x, indices2, output_size=S2)
        x = self.ResUnit4(x + new_x)
        x = self.max_unpool2(x, indices1, output_size=S1)
        x = self.sigmoid(x + x_input)

        x = torch.mul(x_input, x)
        
        return x_input + x
    
class ResNetAtt(nn.Module):
    """
    Model from "Residential Appliance Detection Using Attentionbased Deep Convolutional Neural Network" paper.
    
    Inputs :
    - MTS like [Batch, M, Length]
    - Output : [Batch, nb_class]
    
    Paper authour's parameters :
        - Dilate blocks # 6
        - Dilation rate {1, 2, 4, 8}
        - Attention module # 2
        - Kernel size 8
        - Kernel number 24
        - d_ff (First fully connected before SoftMax) 100
        - Batch size 32
        - Dropout rate 0.1
        - Optimizing method AMSGrad
        - Start learning rate 0.0002
        - Learning rate decay 0.5
        - Training stop early stopping
    """
    def __init__(self, in_channels=1, n_dilated_block=6, n_attention_block=2, in_model_channel=24, 
                 kernel_size=8, dilation_list=[1, 2, 4, 8], d_ff=100, dp_rate=0.1, nb_class=2):
        super(ResNetAtt, self).__init__()
        
        layers = []
        for i in range(n_dilated_block):
            layers.append(DilatedBlock(c_in=in_channels if i==0 else in_model_channel, 
                                       c_out=in_model_channel, kernel_size=kernel_size, 
                                       dilation_list=dilation_list, dp_rate=dp_rate))
            
        self.res_network = torch.nn.Sequential(*layers)    
            
        layers = []
        for i in range(n_attention_block):
            layers.append(AttentionBlock(c_in=in_model_channel, c_out=in_model_channel, 
                                         kernel_size=kernel_size, dp_rate=dp_rate))
            
        self.att_network = torch.nn.Sequential(*layers)
        
        self.linear1 = nn.LazyLinear(d_ff, bias=False) # According to paper detail 500 * 100 = 50 000 parameters so no bias
        self.linear2 = nn.LazyLinear(nb_class)

    def forward(self, x) -> torch.Tensor:
        x = self.res_network(x)
        x = self.att_network(x)
        x = F.relu(self.linear1(x))

        return self.linear2(torch.flatten(x, 1))
