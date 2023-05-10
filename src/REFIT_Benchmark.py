import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.dictionary_based import IndividualBOSS, ContractableBOSS
from sktime.classification.interval_based import TimeSeriesForestClassifier, RandomIntervalSpectralEnsemble, DrCIF
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

sys.path.append(os.getcwd())
from Utils._utils_ import *
from Utils._utils_preprocessing_ import *

from Utils.Models.ResNet import ResNet
from Utils.Models.InceptionTime import Inception, InceptionTime
from Utils.Models.ConvNet import ConvNet
from Utils.Models.ResNetAtt import ResNetAtt

    
def launch_sktime_training(model, X_train, y_train, X_test, y_test, path_res):

    if not check_file_exist(path_res+'.pt'):
        # Equalize class for training
        X_train, y_train = RandomUnderSampler_(X_train, y_train)
        
        sk_trainer = classif_trainer_sktime(model.reset(), verbose=False, save_model=False, 
                                            save_checkpoint=True, path_checkpoint=path_res) 
                                    
        sk_trainer.train(X_train, y_train)
        sk_trainer.evaluate(X_test, y_test)
    
    return

def launch_deep_training(model, X_train, y_train, X_valid, y_valid, X_test, y_test, path_res, max_epochs=100):
    
    # Equalize class for training
    X_train, y_train = RandomUnderSampler_(X_train, y_train)
    
    model_instance = model['instance']

    train_dataset = TSDataset(X_train, y_train)
    valid_dataset = TSDataset(X_valid, y_valid)
    test_dataset  = TSDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

    model_trainer = classif_trainer(model_instance(),
                                    train_loader=train_loader, valid_loader=valid_loader,
                                    learning_rate=model['lr'], weight_decay=model['wd'],
                                    patience_es=20, patience_rlr=5,
                                    device="cuda", all_gpu=True,
                                    verbose=False, plotloss=False, 
                                    save_checkpoint=True, path_checkpoint=path_res)

    model_trainer.train(n_epochs=max_epochs)
    model_trainer.restore_best_weights()
    model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))
    
    return

def REFIT_case(chosen_clf, classifiers, list_dict_case, list_param, path_res, perc_data_used=1, nb_houses_used='all'):

    model = classifiers[chosen_clf]
    
    for dict_case in list_dict_case:
        
        case = dict_case['app']
        dir_res_case = create_dir(path_res+case+os.sep)
        
        for param in list_param:
            
            dir_res_param = create_dir(dir_res_case+param['sampling_rate']+os.sep)

            for seed in range(5):
                np.random.seed(seed=seed)
                
                house_with_app = np.array(dict_case['house_with_app_i'])
                house_without_app = np.array(dict_case['house_without_app_i'])

                # First, to be sure to have at least one house with appliance in train and test
                ind_house_train = list(np.random.choice(house_with_app, size=1, replace=False))
                ind_house_test  = list(np.random.choice(house_without_app, size=1, replace=False))
                
                # Rest of indices for train and test
                rest_indices = np.array(list(house_with_app) + list(house_without_app))
                
                # Get second house indice of for test dataset
                ind_house_test  = ind_house_test + list(np.random.choice(rest_indices, size=1, replace=False))
                
                if nb_houses_used=='all':
                    # Get rest of houses indices for train if all the houses used
                    ind_house_train = ind_house_train + list(rest_indices)
                else:
                    # Get corresponding number of houses indices for train if n houses used
                    ind_house_train = ind_house_train + list(np.random.choice(rest_indices, size=nb_houses_used, replace=False))
                
                databuilder = REFITData(appliance_names=[case],
                                        sampling_rate=param['sampling_rate'],
                                        window_size=param['window_size'],
                                        train_house_indicies=ind_house_train,
                                        test_house_indicies=ind_house_test,
                                        limit_ffill=param['limit_ffill']
                                        )

                train, test = databuilder.BuildSaveNILMDataset(perc=perc_data_used)
                
                np.random.shuffle(train)
                X_train, y_train = TransformNILMDatasettoClassif(train, threshold=dict_case['threshold'])
                X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
                X_test, y_test = TransformNILMDatasettoClassif(test, threshold=dict_case['threshold'])
                
                path_to_save = dir_res_param+chosen_clf+'_'+str(seed)

                # ==================== Ensemble of Inception training =================== #
                if chosen_clf=="Inception":
                    path_inception = create_dir(path_to_save+os.sep)
                    
                    for i in range(5):
                        if not check_file_exist(path_inception+'Inception'+str(i)+'.pt'):
                            launch_deep_training(model, X_train, y_train, X_valid, y_valid, X_test, y_test, path_inception+'Inception'+str(i))

                    launch_classif(InceptionTime(Inception(), path_inception, 5), X_train, y_train, X_test, y_test, path_to_save)
                    
                # ==================== Deep Learning Classifier =================== #
                elif chosen_clf=="ResNet" or chosen_clf=="ConvNet" or chosen_clf=="ResNetAtt":
                    launch_deep_training(model, X_train, y_train, X_valid, y_valid, X_test, y_test, path_to_save)
                    
                # ==================== Sktime Classifier =================== #
                else:
                    launch_classif(model, X_train, y_train, X_test, y_test, path_to_save)

    return
  

if __name__ == "__main__":

    path_res = None # To be fill
    
    #============= Base case with all possible houses =============#
    list_case = [{'app': 'Dishwasher',  
                  'house_with_app_i': [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15, 16, 18, 20, 21],
                  'house_without_app_i': [4, 8, 12, 15, 16, 17, 19],
                  'threshold': 5},
                 {'app': 'Washing Machine',  
                  'house_with_app_i': [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15, 16, 18, 20, 21],
                  'house_without_app_i': [12],
                  'threshold': 5},
                 {'app': 'Kettle',  
                  'house_with_app_i': [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 20],
                  'house_without_app_i': [1, 10, 16, 18],
                  'threshold': 5},
                 {'app': 'Computer Site',  
                  'house_with_app_i': [1, 4, 5, 11, 13, 15, 16, 17, 20],
                  'house_without_app_i': [2, 3, 7,  9, 10, 12, 19, 21],
                  'threshold': 5},
                 {'app': 'Television Site',  
                  'house_with_app_i': [1, 2, 3, 4, 5, 6, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21],
                  'house_without_app_i': [11],
                  'threshold': 5},
                 {'app': 'Tumble Dryer',  
                  'house_with_app_i': [1, 3, 5, 7, 8, 13, 15, 20, 21],
                  'house_without_app_i': [2, 4, 6, 10, 11, 12, 16, 19],
                  'threshold': 5},
                 {'app': 'Microwave',  
                  'house_with_app_i': [2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20],
                  'house_without_app_i': [1, 7, 16],
                  'threshold': 5}
                ]

    # =============== Random convolution =============== #
    clf_arsenal = Arsenal(n_jobs=-1)
    clf_rocket = RocketClassifier(rocket_transform='rocket', n_jobs=-1)
    clf_minirocket = RocketClassifier(rocket_transform='minirocket', n_jobs=-1)
    
    # =============== Random interval =============== #
    clf_tsfc = TimeSeriesForestClassifier(min_interval=10, n_jobs=-1)
    clf_rise = RandomIntervalSpectralEnsemble(n_jobs=-1)
    clf_drcif = DrCIF(n_jobs=-1)
    
    # =============== Dictionnary =============== #
    clf_boss = IndividualBOSS(n_jobs=-1)
    clf_eboss = BOSSEnsemble(n_jobs=-1)
    clf_cboss = ContractableBOSS(n_jobs=-1)

    # =============== Shape distance =============== #
    clf_knne = KNeighborsTimeSeriesClassifier(algorithm='auto', distance='euclidean', n_jobs=-1)
    clf_knndtw = KNeighborsTimeSeriesClassifier(algorithm='auto', distance='dtw', n_jobs=-1)
    
    classifiers = {'Arsenal': clf_arsenal,
                   'Rocket'  : clf_rocket,
                   'Minirocket': clf_minirocket,
                   'TimeSeriesForest': clf_tsfc,
                   'Rise': clf_rise,
                   'DrCIF': clf_drcif,
                   'BOSS': clf_boss,
                   'eBOSS': clf_eboss,
                   'cBOSS': clf_cboss,
                   'KNNeucli' : clf_knne,
                   'KNNdtw' : clf_knndtw,
                   'ResNet': {'model_inst': ResNet, 'batch_size': 32, 'lr': 1e-3, 'wd': 0},
                   'Inception': {'model_inst': Inception, 'batch_size': 32, 'lr': 1e-3, 'wd': 0},
                   'ConvNet': {'model_inst': ConvNet, 'batch_size': 32, 'lr': 1e-3, 'wd': 0},
                   'ResNetAtt': {'model_inst': ResNetAtt, 'batch_size': 32, 'lr': 0.0002, 'wd': 0.5}
                  }

    # ====== List of dict parameters for resampling ===== #
    list_param = [{'sampling_rate': '30T', 'window_size': 48,   'limit_ffill': 2},
                  {'sampling_rate': '15T', 'window_size': 96,   'limit_ffill': 4},
                  {'sampling_rate': '10T', 'window_size': 144,  'limit_ffill': 6},
                  {'sampling_rate': '1T',  'window_size': 1440, 'limit_ffill': 60}]


    # ====== Number of houses use with all the data available ====== #
    nb_houses = [4, 8, 12, 'all']
    for nb_h in nb_houses:
        path_res_n_houses = create_dir(path_res + str(h) + '_houses' + os.sep)
        REFIT_case(str(sys.argv[1]), classifiers, [list_case[int(sys.argv[2])]], list_param, path_res_n_houses, nb_houses_used=nb_h)
        
     # ====== Percentage of data used by houses ====== #        
    percentage_data_h = [20, 40, 60, 80]
    for p in percentage_data_h:
        path_res_n_houses = create_dir(path_res + str(p) + '_perc' + os.sep)
        REFIT_case(str(sys.argv[1]), classifiers, [list_case[int(sys.argv[2])]], list_param, path_res_n_houses, perc_data_used=p/100)
