import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import csv

class FeatureDataset(Dataset):
    def __init__(self, file_name, start_x, end_y):

        #read csv file and load row data into variables
        file_out = pd.read_csv(file_name)
        x = file_out.iloc[:, start_x:].values
        y = file_out.iloc[:, :end_y].values
        print(x, y)

        # Feature Scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # converting to torch tensors
        self.x_train = torch.tensor(x_train, dtype = torch.float32)
        self.y_train = torch.tensor(y_train)

        #print(x_train, y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]