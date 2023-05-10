import os
import numpy as np
import torch
import torch.nn as nn
from Utils._utils_ import TSDataset

def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes

def pass_through(X):
    return X

class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)
    
class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)

    
class InceptionTime():
    """
    InceptionTime implementation class wrap in an sktime model API like.
    All the model of the ensemble must be trained before and stored in a same folder.
    
    Parameters :
        - inception_instance : An InceptionClassifier instance with SAME parameters as trained models.
        - path_models : Path to trained models, each model in a folder and named as "InceptionX" with X indice of the model.
        - n_model : Number of Inception classifier in the ensemble.
        - device : Device for evaluate inception classifiers model (cpu, cuda).
    """
    def __init__(self, inception_instance, path_models, n_model, device="cuda"):
        self.path_models = path_models
        self.n_model = n_model
        self.inception_instance = inception_instance.to(device)
        self.device = device
        self.is_fitted = self._check_isfitted()
    
    def fit(self, X=None, y=None):
        self.is_fitted = self._check_isfitted()
        for i in range(self.n_model):
            if not os.path.isfile(self.path_models+'Inception'+str(i)+'.pt'):
                raise ValueError('File {} not found.'.format(self.path_models+'Inception'+str(i)+'.pt'))
        return
    
    def predict(self, X):
        pt_dataset = TSDataset(X)
        y_pred = np.zeros(len(X))
        
        for key, inst in enumerate(pt_dataset):
            pred_ensemble = np.array([])
            for i in range(self.n_model):
                self.inception_instance.load_state_dict(torch.load(self.path_models+'Inception'+str(i)+'.pt')['model_state_dict'])
                self.inception_instance.eval()
                pred = nn.Softmax(dim=1)(self.inception_instance(torch.Tensor(np.expand_dims(inst, axis=1)).to(self.device))).detach().cpu().numpy()
                pred_ensemble = pred_ensemble + pred if pred_ensemble.size else pred
            pred_ensemble = pred_ensemble / self.n_model
            y_pred[key] = pred_ensemble.argmax()
        return y_pred
    
    def predict_proba(self, X):
        pt_dataset = TSDataset(X)
        y_pred = np.array([])
        
        for key, inst in enumerate(pt_dataset):
            pred_ensemble = np.array([])
            for i in range(self.n_model):
                self.inception_instance.load_state_dict(torch.load(self.path_models+'Inception'+str(i)+'.pt')['model_state_dict'])
                self.inception_instance.eval()
                pred = nn.Softmax(dim=1)(self.inception_instance(torch.Tensor(np.expand_dims(inst, axis=1)).to(self.device))).detach().cpu().numpy()
                pred_ensemble = pred_ensemble + pred if pred_ensemble.size else pred
            pred_ensemble = pred_ensemble / self.n_model
            y_pred = np.concatenate((y_pred, pred_ensemble), axis=0) if y_pred.size else pred_ensemble
        return y_pred
    
    def reset(self):
        return self
    
    def _check_isfitted(self):
        is_fitted=True 
        for i in range(self.n_model):
            if not os.path.isfile(self.path_models+'Inception'+str(i)+'.pt'):
                is_fitted=False
        return is_fitted
    
    
class Inception(nn.Module):
    def __init__(self, in_channels=1, nb_class=2, n_filters=32, n_blocks=2,
                 kernel_sizes=[9, 19, 39], bottleneck_channels=32, pooling_size=1,
                 activation=nn.ReLU(), return_indices=False, use_residual=True):
        super(Inception, self).__init__()
        
        layers = []
        
        layers.append(InceptionBlock(in_channels=in_channels, 
                                     n_filters=n_filters, 
                                     kernel_sizes=kernel_sizes,
                                     bottleneck_channels=bottleneck_channels,
                                     use_residual=use_residual,
                                     activation=activation
                                     ))
                      
        for i in range(n_blocks-1):
            layers.append(InceptionBlock(in_channels=n_filters*(len(kernel_sizes)+1), 
                                         n_filters=n_filters, 
                                         kernel_sizes=kernel_sizes,
                                         bottleneck_channels=bottleneck_channels,
                                         use_residual=use_residual,
                                         activation=activation
                                        ))
        
        self.Blocks = nn.Sequential(*layers)
        
        N = (len(kernel_sizes)+1)*n_filters*pooling_size
        
        self.Pooling = nn.Sequential(nn.AdaptiveAvgPool1d(output_size=pooling_size), Flatten(out_features=N))
        self.Linear = nn.Linear(in_features=N, out_features=nb_class)
                          
    def forward(self, X):
        X = self.Blocks(X)
        X = self.Pooling(X)
        
        return self.Linear(X)

                                 
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, 
                 use_residual=True, activation=nn.ReLU(), return_indices=False):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.inception_1 = InceptionModule(
                            in_channels=in_channels,
                            n_filters=n_filters,
                            kernel_sizes=kernel_sizes,
                            bottleneck_channels=bottleneck_channels,
                            activation=activation,
                            return_indices=return_indices
                            )
        self.inception_2 = InceptionModule(
                            in_channels=(len(kernel_sizes)+1)*n_filters,
                            n_filters=n_filters,
                            kernel_sizes=kernel_sizes,
                            bottleneck_channels=bottleneck_channels,
                            activation=activation,
                            return_indices=return_indices
                            )
        self.inception_3 = InceptionModule(
                            in_channels=(len(kernel_sizes)+1)*n_filters,
                            n_filters=n_filters,
                            kernel_sizes=kernel_sizes,
                            bottleneck_channels=bottleneck_channels,
                            activation=activation,
                            return_indices=return_indices
                            )    
        if self.use_residual:
            self.residual = nn.Sequential(
                                nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=(len(kernel_sizes)+1)*n_filters, 
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                    ),
                                nn.BatchNorm1d(
                                    num_features=(len(kernel_sizes)+1)*n_filters
                                    )
                                )

    def forward(self, X):
        if self.return_indices:
            Z, i1 = self.inception_1(X)
            Z, i2 = self.inception_2(Z)
            Z, i3 = self.inception_3(Z)
        else:
            Z = self.inception_1(X)
            Z = self.inception_2(Z)
            Z = self.inception_3(Z)
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z
                                 

class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], 
                 bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
        """
        : param in_channels          Number of input channels (input features)
        : param n_filters            Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes         List of kernel sizes for each convolution.
                                     Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                     This is nessesery because of padding size.
                                     For correction of kernel_sizes use function "correct_sizes". 
        : param bottleneck_channels  Number of output channels in bottleneck. 
                                     Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation           Activation function for output tensor (nn.ReLU()). 
        : param return_indices       Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
        """
        super(InceptionModule, self).__init__()
        self.return_indices=return_indices
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                                in_channels=in_channels, 
                                out_channels=bottleneck_channels, 
                                kernel_size=1, 
                                stride=1, 
                                bias=False
                                )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[0], 
                                        stride=1, 
                                        padding=kernel_sizes[0]//2, 
                                        bias=False
                                        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[1], 
                                        stride=1, 
                                        padding=kernel_sizes[1]//2, 
                                        bias=False
                                        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[2], 
                                        stride=1, 
                                        padding=kernel_sizes[2]//2, 
                                        bias=False
                                        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=n_filters, 
                                    kernel_size=1, 
                                    stride=1,
                                    padding=0, 
                                    bias=False
                                    )
        self.batch_norm = nn.BatchNorm1d(num_features=(len(kernel_sizes)+1)*n_filters)
        self.activation = activation

    def forward(self, X):
        # step 1
        Z_bottleneck = self.bottleneck(X)
        if self.return_indices:
            Z_maxpool, indices = self.max_pool(X)
        else:
            Z_maxpool = self.max_pool(X)
        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        # step 3 
        Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        Z = self.activation(self.batch_norm(Z))
        if self.return_indices:
            return Z, indices
        else:
            return Z