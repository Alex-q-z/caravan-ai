import copy
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class IntrusionDetectionDataset(Dataset):
    def __init__(self, 
                 file_path, 
                 header=0,
                 standardize=False, 
                 normalize=False, 
                 device='cuda'):
        
        self.df = pd.read_csv(file_path, index_col=None, header=header)
        self.df.columns = [x.strip().lower() for x in self.df.columns]
        self.header_names = self.df.columns.tolist()

        # use numeric columns as features
        self.df_numeric = self.df.select_dtypes(include=['number'])
        self.feature_names = self.df_numeric.columns.tolist()

        self.data = torch.from_numpy(self.df_numeric.values).float().to(device)
        self.features = self.data[:,:-1].type(torch.float32)
        self.labels = self.data[:,-1].type(torch.int32)
                
        # standardize the data if needed
        if standardize:
            standard_scaler = StandardScaler()
            self.features = standard_scaler.fit_transform(self.features)
            self.features = torch.from_numpy(self.features).type(torch.float32)
        
        # normalize the data if needed
        if normalize:
            minmax_scaler = MinMaxScaler()
            self.features = minmax_scaler.fit_transform(self.features)
            self.features = torch.from_numpy(self.features).type(torch.float32)
    
    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]

    def __len__(self):
        return len(self.data)

    # merge another dataset into an existing one
    def merge(self, another_dataset):
        self.df = pd.concat([self.df, another_dataset.df])
        self.df_numeric = pd.concat([self.df_numeric, another_dataset.df_numeric])
        self.data = torch.cat((self.data, another_dataset.data), 0)
        self.features = torch.cat((self.features, another_dataset.features), 0)
        self.labels = torch.cat((self.labels, another_dataset.labels), 0)

    # concat two datasets
    def concat(self, another_dataset):
        new_dataset = copy.deepcopy(self)
        new_dataset.df = pd.concat([self.df, another_dataset.df])
        new_dataset.df_numeric = pd.concat([self.df_numeric, another_dataset.df_numeric])
        new_dataset.data = torch.cat((self.data, another_dataset.data), 0)
        new_dataset.features = torch.cat((self.features, another_dataset.features), 0)
        new_dataset.labels = torch.cat((self.labels, another_dataset.labels), 0)
        return new_dataset