import pandas as pd
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, file_name, stopsign, mode = 'train'):
        file_out = pd.read_csv(file_name)
        if mode == 'train':
            self.x = file_out.iloc[:, stopsign:].values
            self.y = file_out.iloc[:, 1:stopsign].values
        else: 
            self.x = file_out.iloc[:, stopsign].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input = torch.Tensor(self.x[idx])
        output = torch.Tensor(self.y[idx])

        return {
            'input': input,
            'output': output,
        }