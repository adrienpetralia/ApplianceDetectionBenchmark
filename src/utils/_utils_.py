import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

    
class TSDataset(torch.utils.data.Dataset):
    """
    MAP-Style PyTorch Time series Dataset with possibility of scaling
    
    - X matrix of TS input, can be 2D or 3D, Dataframe instance or Numpy array instance.
    - Labels : y labels associated to time series for classification. Possible to be None.
    - scaler : provided type of scaler (sklearn StandardScaler, MinMaxScaler instance for example).
    - scale_dim : list of dimensions to be scaled in case of multivariate TS.
    """
    def __init__(self, X, labels=None, scaler=StandardScaler(), scale_dim=None):
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(labels, pd.core.frame.DataFrame):
            labels = labels.values
        
        if scaler is not None:
            # ==== Multivariate case ==== #
            if len(X.shape)==3:
                self.scaler_list = []
                self.samples = X
                if scale_dim is None:                    
                    for i in range(X.shape[1]):
                        self.scaler_list.append(StandardScaler())
                        self.samples[:, i, :] = self.scaler_list[i].fit_transform(X[:,i,:].T).T.astype(np.float32)
                else:
                    for idsc, i in enumerate(scale_dim):
                        self.scaler_list.append(StandardScaler())
                        self.samples[:, i, :] = self.scaler_list[idsc].fit_transform(X[:,i,:].T).T.astype(np.float32)
                        
            # ==== Univariate case ==== #
            else:
                self.samples = scaler.fit_transform(X.T).T.astype(np.float32)
        else:
            self.samples = X
            
        if len(self.samples.shape)==2:
            self.samples = np.expand_dims(self.samples, axis=1)
        
        if labels is not None:
            self.labels = labels.ravel()
            assert len(self.samples)==len(self.labels), f"Number of X sample {len(self.samples)} doesn't match number of y sample {len(self.labels)}."
        else:
            self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        if self.labels is None:
            return self.samples[idx]
        else:
            return self.samples[idx], self.labels[idx]


class getmetrics():
    """
    Basics metrics for imbalance classification
    """
    def __init__(self, minority_class=None):
        self.minority_class = minority_class
        
    def __call__(self, y, y_hat):
        metrics = {}

        if self.minority_class is not None:
            minority_class = self.minority_class                
        else:
            y_label = np.unique(y)

            if np.count_nonzero(y==y_label[0]) > np.count_nonzero(y==y_label[1]):
                minority_class = y_label[1]
            else :
                minority_class = y_label[0]

        metrics['ACCURACY'] = accuracy_score(y, y_hat)
        
        metrics['PRECISION'] = precision_score(y, y_hat, pos_label=minority_class, average='binary')
        metrics['RECALL'] = recall_score(y, y_hat, pos_label=minority_class, average='binary')
        metrics['PRECISION_MACRO'] = precision_score(y, y_hat, average='macro')
        metrics['RECALL_MACRO'] = recall_score(y, y_hat, average='macro')
        
        metrics['F1_SCORE'] = f1_score(y, y_hat, pos_label=minority_class, average='binary')
        metrics['F1_SCORE_MACRO'] = f1_score(y, y_hat, average='macro')
        metrics['F1_SCORE_WEIGHTED'] = f1_score(y, y_hat, average='weighted')
        
        metrics['ROC_AUC_SCORE'] = roc_auc_score(y, y_hat)
        metrics['ROC_AUC_SCORE_MACRO'] = roc_auc_score(y, y_hat, average='macro')
        metrics['ROC_AUC_SCORE_WEIGHTED'] = roc_auc_score(y, y_hat, average='weighted')
        
        metrics['CONFUSION_MATRIX'] = confusion_matrix(y, y_hat)

        return metrics    

class classif_trainer_deep():
    def __init__(self,
                 model, 
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=1e-3,
                 criterion=nn.CrossEntropyLoss(),
                 patience_es=None, patience_rlr=None,
                 device="cuda", all_gpu=False,
                 valid_criterion=None,
                 n_warmup_epochs=0,
                 f_metrics=getmetrics(),
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_checkpoint=False, path_checkpoint=None):
        """
        PyTorch Model Trainer Class for classification case
        """

        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.f_metrics = f_metrics
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        
        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion
        
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'
            
        if self.patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                        patience=self.patience_rlr, 
                                                                        verbose=self.verbose,
                                                                        eps=1e-7)
  
        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.voter_time = 0
        self.passed_epochs = 0
        self.best_loss = np.Inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []
               
        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module =========== #
            self.model.to("cpu")
            for ts, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        
        tmp_time = time.time()
        
        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            train_loss, train_accuracy = self.__train()
            self.loss_train_history.append(train_loss)
            self.accuracy_train_history.append(train_accuracy)
            if self.valid_loader is not None:
                valid_loss, valid_accuracy = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
                self.accuracy_valid_history.append(valid_accuracy)
            else:
                valid_loss = train_loss
                
            # =======================reduce lr======================= #
            if self.patience_rlr:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if self.passed_epochs > self.n_warmup_epochs: # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        #flag_es  = 1
                        es_epoch = epoch+1
                        self.passed_epochs+=1
                        if self.verbose:
                            print('Early stopping after {} epochs !'.format(epoch+1))
                        break
        
            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch+1, n_epochs))
                print('    Train loss : {:.4f}, Train acc : {:.2f}%'
                          .format(train_loss, train_accuracy*100))
                
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.4f}, Valid  acc : {:.2f}%'
                              .format(valid_loss, valid_accuracy*100))

            # =======================save log======================= #
            if valid_loss <= self.best_loss and self.passed_epochs>=self.n_warmup_epochs:
                self.best_loss = valid_loss
                self.log = {'valid_metrics': valid_accuracy if self.valid_loader is not None else train_accuracy,
                            'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss_train_history': self.loss_train_history,
                            'loss_valid_history': self.loss_valid_history,
                            'accuracy_train_history': self.accuracy_train_history,
                            'accuracy_valid_history': self.accuracy_valid_history,
                            'value_best_loss': self.best_loss,
                            'epoch_best_loss': self.passed_epochs,
                            'time_best_loss': round((time.time() - tmp_time), 3),
                            }
                if self.save_checkpoint:
                    self.save()
                
            self.passed_epochs+=1
                    
        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()
            
        if self.save_checkpoint:
            self.log['best_model_state_dict'] = torch.load(self.path_checkpoint+'.pt')['model_state_dict']
        
        # =======================update log======================= #
        self.log['training_time'] = self.train_time
        self.log['loss_train_history'] = self.loss_train_history
        self.log['loss_valid_history'] = self.loss_valid_history
        self.log['accuracy_train_history'] = self.accuracy_train_history
        self.log['accuracy_valid_history'] = self.accuracy_valid_history
        
        if self.save_checkpoint:
            self.save()
        return
    
    def evaluate(self, test_loader, mask='test_metrics', return_output=False):
        """
        Public function : model evaluation on test dataset
        """
        tmp_time = time.time()
        mean_loss_eval = []
        y = np.array([])
        y_hat = np.array([])
        with torch.no_grad():
            for ts, labels in test_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = Variable(ts.float()).to(self.device)
                labels = Variable(labels.float()).to(self.device)
                # ===================forward===================== #
                logits = self.model(ts)
                loss = self.valid_criterion(logits.float(), labels.long())
                # =================concatenate=================== #
                _, predicted = torch.max(logits, 1)
                mean_loss_eval.append(loss.item())
                y_hat = np.concatenate((y_hat, predicted.detach().cpu().numpy())) if y_hat.size else predicted.detach().cpu().numpy()
                y = np.concatenate((y, torch.flatten(labels).detach().cpu().numpy())) if y.size else torch.flatten(labels).detach().cpu().numpy()
                
        metrics = self.__apply_metrics(y, y_hat)
        self.eval_time = round((time.time() - tmp_time), 3)
        self.log['eval_time'] = self.eval_time
        self.log[mask] = metrics
        
        if self.save_checkpoint:
            self.save()
        
        if return_output:
            return np.mean(mean_loss_eval), metrics, y, y_hat
        else:
            return np.mean(mean_loss_eval), metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g['lr'] = new_lr
        return
            
    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log['best_model_state_dict'])
            else:
                self.model.load_state_dict(self.log['best_model_state_dict'])
            print('Restored best model met during training.')
        except KeyError:
            print('Error during loading log checkpoint state dict : no update.')
        return
    
    def __train(self):
        """
        Private function : model training loop over data loader
        """
        total_sample_train = 0
        mean_loss_train = []
        mean_accuracy_train = []
        
        for ts, labels in self.train_loader:
            self.model.train()
            # ===================variables=================== #
            ts = Variable(ts.float()).to(self.device)
            labels = Variable(labels.float()).to(self.device)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            logits = self.model(ts)
            # ===================backward==================== #
            loss_train = self.train_criterion(logits.float(), labels.long())
            loss_train.backward()
            self.optimizer.step()
            # ================eval on train================== #
            total_sample_train += labels.size(0)
            _, predicted_train = torch.max(logits, 1)
            correct_train = (predicted_train.to(self.device) == labels.to(self.device)).sum().item()
            mean_loss_train.append(loss_train.item())
            mean_accuracy_train.append(correct_train)
            
        return np.mean(mean_loss_train), np.sum(mean_accuracy_train)/total_sample_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        total_sample_valid = 0
        mean_loss_valid = []
        mean_accuracy_valid = []
        
        with torch.no_grad():
            for ts, labels in self.valid_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = Variable(ts.float()).to(self.device)
                labels = Variable(labels.float()).to(self.device)
                logits = self.model(ts)
                loss_valid = self.valid_criterion(logits.float(), labels.long())
                # ================eval on test=================== #
                total_sample_valid += labels.size(0)
                _, predicted = torch.max(logits, 1)
                correct = (predicted.to(self.device) == labels.to(self.device)).sum().item()
                mean_loss_valid.append(loss_valid.item())
                mean_accuracy_valid.append(correct)

        return np.mean(mean_loss_valid), np.sum(mean_accuracy_valid)/total_sample_valid
    
    def __apply_metrics(self, y, y_hat):
        """
        Private function : apply provided metrics
        
        !!! Provided metric function must be callable !!!
        """
            
        return self.f_metrics(y, y_hat)

    
class classif_trainer_sktime():
    def __init__(self,
                 model,
                 f_metrics=getmetrics(),
                 verbose=True, save_model=False,
                 save_checkpoint=False, path_checkpoint=None):
        """
        Trainer designed for scikit API like model and classification cases
        """
        self.model = model
        self.f_metrics = f_metrics
        self.verbose = verbose
        self.save_model = save_model
        self.save_checkpoint = save_checkpoint
        
        if path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'
        
        self.train_time = 0
        self.test_time = 0
        self.log = {}
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None, scaler=StandardScaler()):
        """
        Public function : fit API call
        
        -> Z-normalization of the data by default
        """
        
        if scaler is not None:
            X_train = scaler.fit_transform(X_train.T).T
        
        _t = time.time()
        self.model.fit(X_train, y_train.ravel())
        self.train_time = round((time.time() - _t), 3)
        self.log['training_time'] = self.train_time
        
        if self.save_model:
            self.log['model'] = self.model
            
        if X_valid is not None and y_valid is not None:
            if scaler is not None:
                X_valid = scaler.fit_transform(X_valid.T).T
            valid_metrics = self.evaluate(X_valid, y_valid, mask='valid_metrics')
            if self.verbose:
                print('Valid metrics :', valid_metrics)
        
        if self.verbose:
            print('Training time :', self.train_time)
        
        return
    
    def evaluate(self, X_test, y_test, mask='test_metrics', scaler=StandardScaler()):
        """
        Public function : predict API call then evaluation with given metric function
        
        -> Z-normalization of the data by default
        """
        
        if scaler is not None:
            X_test = scaler.fit_transform(X_test.T).T
        
        _t = time.time()
        metrics = self.__apply_metrics(y_test.ravel(), self.model.predict(X_test))
        self.log[mask] = metrics
        self.test_time = round((time.time() - _t), 3)
        self.log['eval_time'] = self.test_time
        
        if self.save_checkpoint:
            self.save()

        return metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def __apply_metrics(self, y, y_hat):
        """
        Private function : apply provided metrics
        
        !!! Provided metric function must be callable !!!
        """
        
        return(self.f_metrics(y, y_hat))
    
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False