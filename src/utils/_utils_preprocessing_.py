import os
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler


#####################################################################################################################
# Utils for data preprocessing
#####################################################################################################################

def create_dir(path):
    try:
        os.mkdir(path)
    except:
        pass
    return path

def check_file_exist(path):
    return os.path.isfile(path)


def RandomUnderSampler_(X, y, sampling_strategy='auto', seed=0):
    np.random.seed(seed)
    X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
    Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
    np.random.shuffle(Mat)
    Mat = Mat.astype(np.float32)
        
    return Mat[:, :-1], Mat[:, -1]


def load_transpose_CER(file_x, file_case=None, mask='id_pdl', scale_data=False, scaler=StandardScaler()):

    df_data_x = pd.read_csv(file_x, low_memory=False).set_index('index')
    df_data_x.index.name = None
    df_data_x = df_data_x.T
    df_data_x[mask] = df_data_x[mask].astype('int32')
    df_data_x = df_data_x.set_index(mask)
    
    if scale_data:
        df_data_x = pd.DataFrame(scaler.fit_transform(df_data_x.T).T, columns=df_data_x.columns, index=df_data_x.index)
        
    if file_case is not None:
        case = pd.read_csv(file_case).set_index(mask)
        df_data = pd.merge(df_data_x, case, on=mask)
    else:
        df_data = df_data_x.set_index(mask)
    
    return df_data
        
        
def split_train_valid_test_pdl(df_data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0):

    np.random.seed(seed)
    list_pdl = np.array(df_data.index.unique())
    np.random.shuffle(list_pdl)
    pdl_train_valid = list_pdl[:int(len(list_pdl) * (1-test_size))]
    pdl_test = list_pdl[int(len(list_pdl) * (1-test_size)):]
    np.random.shuffle(pdl_train_valid)
    pdl_train = pdl_train_valid[:int(len(pdl_train_valid) * (1-valid_size))]
    
    df_train = df_data.loc[pdl_train, :].copy()
    df_test = df_data.loc[pdl_test, :].copy()
    

    df_train = df_train.sample(frac=1, random_state=seed)
    df_test = df_test.sample(frac=1, random_state=seed)

    X_train = df_train.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
    y_train = df_train.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
    X_test = df_test.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
    y_test = df_test.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
    
    if valid_size != 0:
        pdl_valid = pdl_train_valid[int(len(pdl_train_valid) * (1-valid_size)):]
        df_valid = df_data.loc[pdl_valid, :].copy()
        df_valid = df_valid.sample(frac=1, random_state=seed)
        X_valid = df_valid.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_valid = df_valid.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test
            
            
class NILMData(object):
    def __init__(self, 
                 data_path,
                 train_house_indicies,
                 appliance_names,
                 sampling_rate,
                 window_size,
                 limit_ffill,
                 cutoff,
                 window_stride=None,
                 test_house_indicies=None,
                 save_path=None):
        
        # =============== Class variables =============== #
        self.data_path = data_path
        self.save_path = save_path
        self.house_train_indicies = train_house_indicies
        self.house_test_indicies = test_house_indicies
        self.appliance_names = appliance_names
        self.sampling_rate = sampling_rate 
        self.limit_ffill = limit_ffill
        self.window_size = window_size
        self.cutoff = cutoff
        
        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size
            
        # ======= Check ID and Appliances names ======= #
        self._check_house_ids()
        self._check_appliance_names()
        
    def BuildSaveNILMDataset(self, mask_index='Time'):
        """
        -> Call GetNILMDataset for train (and test indicies if provided).
        -> Save train and test if save_path is provided
        
        Return : - train : 4D numpy.ndarray
                 or
                 - tuple : train : 4D numpy.ndarray
                           test : 4D numpy.ndarray
        """
        train = self.GetNILMDataset(self.house_train_indicies, mask_index=mask_index)
        if self.save_path is not None:
            torch.save(torch.Tensor(train), self.save_path+'TrainData.pt')
            
        if self.house_test_indicies is not None:
            test = self.GetNILMDataset(self.house_test_indicies, mask_index=mask_index)
            if self.save_path is not None:
                torch.save(torch.Tensor(test), self.save_path+'TestData.pt')
            
            return train, test
        else:
            return train
            
    def GetNILMDataset(self, house_indicies, mask_index='Time'):
        """
        Process data to build NILM usecase
        
        Return : np.ndarray of size [N_ts, M_appliances, 2, Win_Size]
        
        -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
        -2nd dimension : nb chosen appliances
                        -> indice 0 for aggregate load curve
                        -> Other appliance in same order as given "appliance_names" list
        -3rd dimension : access to load curve (values of consumption in Wh) or states of activation 
                         of the appliance (0 or 1 for each time step)
                        -> indice 0 : access to load curve
                        -> indice 1 : access to states of activation (0 or 1 for each time step)
        -4th dimension : values
        """
        
        output_data = np.array([])
        
        for indice in house_indicies:
            data = self._get_dataframe(indice)
            data[data < 5] = 0 # Remove small value
            stems = self._get_stems(data)
            
            if self.window_size==self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)
            
            X = np.empty((len(house_indicies) * n_wins, len(self.appliance_names), 2, self.window_size))
            
            cpt = 0
            for i in range(n_wins):
                tmp = stems[:, i*self.window_stride:i*self.window_stride+self.window_size]
                if not self._check_anynan(tmp[0, :]):
                    for j in range(len(self.appliance_names)):
                        X[cpt, j, 0, :] = tmp[j, :]
                        X[cpt, j, 1, :] = (tmp[j, :] > 0).astype(dtype=int)
                    cpt += 1

            output_data = np.concatenate((output_data, X[:cpt, :, :, :]), axis=0) if output_data.size else X[:cpt, :, :, :]
                        
        return output_data
    
    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.
        
        Return : np.ndarray instance
        """
        stems = np.empty((len(self.appliance_names), dataframe.shape[0]))
        for key, names in enumerate(self.appliance_names):
            if names in dataframe:
                stems[key, :] = np.clip(dataframe[names].values, a_min=0, a_max=self.cutoff)
            else:
                stems[key, :] = np.zeros(dataframe.shape[0])
        return stems
            
    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return
    
    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.dot(a, a))
    
    
class REFITData(NILMData):
    def __init__(self, 
                 train_house_indicies,
                 appliance_names,
                 sampling_rate,
                 window_size,
                 limit_ffill,
                 data_path=os.getcwd()+'/Datasets/REFIT/RAW_DATA_CLEAN/',
                 cutoff=6000,
                 window_stride=None,
                 save_path=None,
                 test_house_indicies=None):
        super().__init__(data_path=data_path, 
                         save_path=save_path, 
                         train_house_indicies=train_house_indicies, 
                         test_house_indicies=test_house_indicies, 
                         appliance_names=appliance_names,
                         sampling_rate=sampling_rate, 
                         window_size=window_size, 
                         window_stride=window_stride, 
                         limit_ffill=limit_ffill, 
                         cutoff=cutoff)
        
        # ======= Add aggregate to appliance(s) list ======= #
        self.appliance_names = ['Aggregate'] + self.appliance_names
        assert data_path is not None, f"Provide path to UKDALE dataset."
    
    def _get_dataframe(self, indice):
        """
        Load house data and rename columns with appliances names.
        
        Return : pd.core.frame.DataFrame instance
        """
        file = self.data_path+'CLEAN_House'+str(indice)+'.csv'
        self._check_if_file_exist(file)
        labels_houses = pd.read_csv(self.data_path+'HOUSES_Labels').set_index('House_id')
        dataframe = pd.read_csv(file)
        dataframe.columns = list(labels_houses.loc[int(indice)].values)
        dataframe = dataframe.set_index('Time').sort_index()
        dataframe.index = pd.to_datetime(dataframe.index)
        idx_to_drop = dataframe[dataframe['Issues']==1].index
        dataframe = dataframe.drop(index=idx_to_drop, axis=0)
        dataframe = dataframe.resample(rule=self.sampling_rate).mean().ffill(limit=self.limit_ffill)
        return dataframe
    
    def _check_house_ids(self):
        """
        Check houses indicies for REFIT Dataset.
        """
        if self.house_test_indicies is not None:
            house_indicies = self.house_train_indicies + self.house_test_indicies
        else:
            house_indicies = self.house_train_indicies
        for house_id in house_indicies:
            assert house_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21], f"House indice unknow for REFIT Dataset, got: {house_id}"
        return
    
    def _check_appliance_names(self):
        """
        Check appliances names for REFIT case.
        """
        for appliance in self.appliance_names:
            assert appliance in ['Fridge-Freezer', 'Fridge', 'Freezer', 'Electric Heater', 'Washing Machine', 'Computer Site', 'Television Site', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle'], f"Selected applicance unknow for REFIT Dataset, got: {appliance}"
        return
    
    
class UKDALEData(NILMData):
    def __init__(self, 
                 train_house_indicies,
                 appliance_names,
                 sampling_rate,
                 window_size,
                 limit_ffill,
                 data_path=os.getcwd()+'/Datasets/UKDALE/',
                 cutoff=6000,
                 window_stride=None,
                 save_path=None,
                 test_house_indicies=None):
        super().__init__(data_path=data_path, 
                         save_path=save_path, 
                         train_house_indicies=train_house_indicies, 
                         test_house_indicies=test_house_indicies, 
                         appliance_names=appliance_names,
                         sampling_rate=sampling_rate, 
                         window_size=window_size, 
                         window_stride=window_stride, 
                         limit_ffill=limit_ffill, 
                         cutoff=cutoff)
        
        # ======= Add aggregate to appliance(s) list ======= #
        self.appliance_names = ['aggregate'] + appliance_names
        
        assert data_path is not None, f"Provide path to UKDALE dataset."
    
    def _get_dataframe(self, indice):
        """
        Load houses data and return one dataframe with aggregate and appliance resampled at chosen time step.
        
        Return : pd.core.frame.DataFrame instance
        """
        path_house = self.data_path+'House'+str(indice)+os.sep
        self._check_if_file_exist(path_house+'labels.dat') # Check if labels exist at provided path
        
        # House labels
        house_label = pd.read_csv(path_house+'labels.dat',    sep=' ', header=None)
        house_label.columns = ['id', 'appliance_name']
        
        # Aggregate and resampling
        house_data = pd.read_csv(path_house+'channel_1.dat', sep=' ', header=None)
        house_data.columns = ['time','aggregate']
        house_data['time'] = pd.to_datetime(house_data['time'], unit = 's')
        house_data = house_data.set_index('time').resample(self.sampling_rate).mean().fillna(method='ffill', limit=self.limit_ffill)
        
        for appliance in self.appliance_names:
            if appliance=='aggregate':
                continue
                
            if len(house_label.loc[house_label['appliance_name']==appliance]['id'].values) != 0:
                i = house_label.loc[house_label['appliance_name']==appliance]['id'].values[0]
                appl_data = pd.read_csv(path_house+'channel_'+str(i)+'.dat', sep=' ', header=None)
                appl_data.columns = ['time',appliance]
                appl_data['time'] = pd.to_datetime(appl_data['time'],unit = 's')
                appl_data = appl_data.set_index('time').resample(self.sampling_rate).mean().fillna(method = 'ffill', limit=self.limit_ffill)   
                house_data = pd.merge(house_data, appl_data, how='inner', on='time')
        
        return house_data
    
    def _check_house_ids(self):
        """
        Check houses indicies for UKDALE Dataset.
        """
        if self.house_test_indicies is not None:
            house_indicies = self.house_train_indicies + self.house_test_indicies
        else:
            house_indicies = self.house_train_indicies
        for house_id in house_indicies:
            assert house_id in [1, 2, 3, 4, 5], f"House indice unknow for UKDALE Dataset, got: {house_id}"
        return
    
    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        for appliance in self.appliance_names:
            assert appliance in ['washing_machine', 'cooker', 'dishwasher', 'kettle', 'fridge', 'microwave', 'electric_heater'], f"Selected applicance unknow for UKDALE Dataset, got: {appliance}"
        return
    
    
def TransformNILMDatasettoClassif(data, threshold=0):
    
    if isinstance(data, torch.Tensor):
            data = data.numpy()
    
    X = np.empty((data.shape[0], data.shape[-1]))
    y = np.zeros(len(data))
    
    for i in range(len(data)):
        X[i] = data[i, 0, 0, :]
        
        if True in (data[i, 1, 0, :] > threshold):
            y[i] = 1
            
    return X, y