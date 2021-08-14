from itertools import count
import numpy as np
import pandas as pd
# from torch.utils import data
# import seaborn as sns
# from tqdm.notebook import tqdm
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split

from os import listdir, stat
from os.path import isfile, join

#files_saccades = [f for f in listdir("window_SACCADES") if isfile(join("window_SACCADES", f))]
files_SP= [f for f in listdir("window_SP_short") if isfile(join("window_SP_short", f))]

# visualize class distribution in train, val, test SACCADES X ONLY
def get_class_distribution_saccades_xdir(obj):
    count_dict = {
        "0": 0,
        "10": 0,
        "20": 0,
        "40": 0,
        "-10": 0,
        "-20": 0,
        "-40":0,
    }
    for i in obj:
        if i == 0: 
            count_dict['0'] += 1
        elif i == 1: 
            count_dict['10'] += 1
        elif i == 2: 
            count_dict['20'] += 1
        elif i == 3: 
            count_dict['40'] += 1
        elif i == 4: 
            count_dict['-10'] += 1  
        elif i == 5: 
            count_dict['-20'] += 1        
        elif i == 6: 
            count_dict['-40'] += 1            
        else:
            print("Check classes.")
    return count_dict

# visualize class distribution in train, val, test SACCADES Y ONLY
def get_class_distribution_saccades_ydir(obj):
    count_dict = {
        "0": 0,
        "5": 0,
        "10": 0,
        "20": 0,
        "-5": 0,
        "-10": 0,
        "-20":0,
    }
    for i in obj:
        if i == 0: 
            count_dict['0'] += 1
        elif i == 1: 
            count_dict['5'] += 1
        elif i == 2: 
            count_dict['10'] += 1
        elif i == 3: 
            count_dict['20'] += 1
        elif i == 4: 
            count_dict['-5'] += 1  
        elif i == 5: 
            count_dict['-10'] += 1        
        elif i == 6: 
            count_dict['-20'] += 1            
        else:
            print("Check classes.")
    return count_dict

# visualize class distribution in train, val, test SP ZONES ONLY
def get_class_distribution_sp_zone(obj):
    count_dict = {
        "border": 0,
        "topright": 0,
        "topleft": 0,
        "bottomright": 0,
        "bottomleft": 0,
    }
    for i in obj:
        if i == 0: 
            count_dict['border'] += 1
        elif i == 1: 
            count_dict['topright'] += 1
        elif i == 2: 
            count_dict['topleft'] += 1
        elif i == 3: 
            count_dict['bottomright'] += 1
        elif i == 4: 
            count_dict['bottomleft'] += 1            
        else:
            print("Check classes.")
    return count_dict

# visualize class distribution in train, val, test SP DIR ONLY
def get_class_distribution_sp_dir(obj):
    count_dict = {
        "moveright": 0,
        "movetopright": 0,
        "movebottomright": 0,
        "moveleft": 0,
        "movetopleft": 0,
        "movebottomleft": 0,
        "standstill":0,
        "movetop": 0,
        "movebottom":0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict['moveright'] += 1
        elif i == 1: 
            count_dict['movetopright'] += 1
        elif i == 2: 
            count_dict['movebottomright'] += 1
        elif i == 3: 
            count_dict['moveleft'] += 1
        elif i == 4: 
            count_dict['movetopleft'] += 1  
        elif i == 5: 
            count_dict['movebottomleft'] += 1        
        elif i == 6: 
            count_dict['standstill'] += 1      
        elif i == 7: 
            count_dict['movetop'] += 1        
        elif i == 8: 
            count_dict['movebottom'] += 1         
        else:
            print("Check classes.")
    return count_dict

# create output mapping for saccades set XDIR ONLY
def outputMapSaccadesXDIR(df):
    classXDIR2idx_saccades = {
        0:0,
        10:1,
        20:2,
        40:3,
        -10:4,
        -20:5,
        -40:6
    }
    CLASSNUM = len(classXDIR2idx_saccades)
    idx2classXDIR_saccades = {v: k for k,v in classXDIR2idx_saccades.items()}
    df['0'].replace(classXDIR2idx_saccades, inplace =True)

    # # create input and output data SACCADES XDIR ONLY!!!
    X = df.iloc[:, 2:]
    y = df.iloc[:, :1]

    # split SACCADES set into Train and Test set for y dir 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=69)
        
    # normalize input between 0 and 1
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    count_dict = get_class_distribution_saccades_xdir(y_train)
    
    return X, y, X_train, X_test, y_train, y_test, CLASSNUM, count_dict

# create output mapping for saccades set YDIR ONLY
def outputMapSaccadesYDIR(df):
    classYDIR2idx_saccades = {
        0:0,
        5:1,
        10:2,
        20:3,
        -5:4,
        -10:5,
        -20:6
    }
    CLASSNUM = len(classYDIR2idx_saccades)
    idx2classYDIR_saccades = {v: k for k,v in classYDIR2idx_saccades.items()}
    df['0.1'].replace(classYDIR2idx_saccades, inplace =True)

    # create input and output data SACCADES YDIR ONLY!!!
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1:2]

    # split SACCADES set into Train and Test set for y dir 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=69)

    # normalize input between 0 and 1
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    count_dict = get_class_distribution_saccades_ydir(y_train)
    
    return X, y, X_train, X_test, y_train, y_test, CLASSNUM, count_dict

# create output mapping for smooth pursuit set ZONE ONLY
def outputMapSPZone(df):
    classZONE2idx_sp = {
        "border":0,
        "topright":1,
        "topleft":2,
        "bottomright":3,
        "bottomleft":4
    }
    CLASSNUM = len(classZONE2idx_sp)
    idx2classZONE_sp = {v: k for k,v in classZONE2idx_sp.items()}
    df['border'].replace(classZONE2idx_sp, inplace =True)

    # create input and output data SP ZONE ONLY!!!
    X = df.iloc[:, 2:]
    y = df.iloc[:, :1]

    # split SP set into Train and Test set for ZONES 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=69)
        
    # normalize input between 0 and 1
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    count_dict = get_class_distribution_sp_zone(y_train)

    return X, y, X_train, X_test, y_train, y_test, CLASSNUM, count_dict

# create output mapping for smooth pursuit set DIR ONLY
def outputMapSPDir(df):
    classDIR2idx_sp = {
        "moveright":0,
        "movetopright":1,
        "movebottomright":2,
        "moveleft":3,
        "movetopleft":4,
        "movebottomleft":5,
        "standstill":6,
        "movetop":7,
        "movebottom":8
    }
    CLASSNUM = len(classDIR2idx_sp)
    idx2classDIR_sp = {v: k for k,v in classDIR2idx_sp.items()}
    df['standstill'].replace(classDIR2idx_sp, inplace =True)

    # create input and output data SP DIR ONLY!!!
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1:2]

    # split SP set into Train and Test set for DIR
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=69)
        
    # normalize input between 0 and 1
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    count_dict = get_class_distribution_sp_dir(y_train)

    return X, y, X_train, X_test, y_train, y_test, CLASSNUM, count_dict

# define custom dataset class
class classifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

# define neural net architecture
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class, num_dimfactor):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, num_feature*num_dimfactor)
        self.layer_2 = nn.Linear(num_feature*num_dimfactor, num_feature)
        self.layer_3 = nn.Linear(num_feature, num_class)
        
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(num_feature*num_dimfactor)
        self.batchnorm2 = nn.BatchNorm1d(num_feature)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.layer_3(x)
        
        return x

# define function for calculating accuracy
def multi_acc(y_pred, y_test, epoch = 0):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    # if epoch == 9:
    #     print(y_pred)
    #     print(y_pred_softmax)
    #     print(y_pred_tags)
    #     print(y_test)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)

    return acc

# define training loop
def train_loop(train_loader, model, criterion, optimizer, epoch):
    print("TRAINING")
    i = 0
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    size = len(train_loader)
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch.squeeze(1))
        train_acc = multi_acc(y_train_pred, y_train_batch.squeeze(1), epoch = epoch)

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc
        i+=1
    train_epoch_loss /= size
    train_epoch_acc /= size

    print(f"loss: {train_epoch_loss:>7f} accuracy: {train_epoch_acc:>7f} [{i:>5d}/{size:>5d}]")
    return train_epoch_loss, train_epoch_acc

# define test loop
def test_loop(test_loader, model, criterion):
    print("TESTING")
    model.eval()
    size = len(test_loader)
    test_epoch_loss = 0 
    test_epoch_acc = 0

    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            y_test_pred = model(X_test_batch)
            
            test_loss = criterion(y_test_pred, y_test_batch.squeeze(1))
            test_acc = multi_acc(y_test_pred, y_test_batch)

            test_epoch_loss += test_loss.item()
            test_epoch_acc += test_acc
    test_epoch_loss /= size
    test_epoch_acc /= size

    print(f"loss: {test_epoch_loss:>7f} accuracy: {test_epoch_acc:>7f}")
    return test_epoch_loss, test_epoch_acc

# dictionary for tracking stats over run

stats_dict = {
    "name" : [],

    "epoch1_train_loss" : [],
    "epoch1_train_acc" : [],
    "epoch1_test_loss" : [],
    "epoch1_test_acc" : [],

    "epoch2_train_loss" : [],
    "epoch2_train_acc" : [],
    "epoch2_test_loss" : [],
    "epoch2_test_acc" : [],

    "epoch3_train_loss" : [],
    "epoch3_train_acc" : [],
    "epoch3_test_loss" : [],
    "epoch3_test_acc" : [],

    "epoch4_train_loss" : [],
    "epoch4_train_acc" : [],
    "epoch4_test_loss" : [],
    "epoch4_test_acc" : [],

    "epoch5_train_loss" : [],
    "epoch5_train_acc" : [],
    "epoch5_test_loss" : [],
    "epoch5_test_acc" : [],

    "epoch6_train_loss" : [],
    "epoch6_train_acc" : [],
    "epoch6_test_loss" : [],
    "epoch6_test_acc" : [],

    "epoch7_train_loss" : [],
    "epoch7_train_acc" : [],
    "epoch7_test_loss" : [],
    "epoch7_test_acc" : [],

    "epoch8_train_loss" : [],
    "epoch8_train_acc" : [],
    "epoch8_test_loss" : [],
    "epoch8_test_acc" : [],

    "epoch9_train_loss" : [],
    "epoch9_train_acc" : [],
    "epoch9_test_loss" : [],
    "epoch9_test_acc" : [],

    "epoch10_train_loss" : [],
    "epoch10_train_acc" : [],
    "epoch10_test_loss" : [],
    "epoch10_test_acc" : [],

    "epoch11_train_loss" : [],
    "epoch11_train_acc" : [],
    "epoch11_test_loss" : [],
    "epoch11_test_acc" : [],

    "epoch12_train_loss" : [],
    "epoch12_train_acc" : [],
    "epoch12_test_loss" : [],
    "epoch12_test_acc" : [],

    "epoch13_train_loss" : [],
    "epoch13_train_acc" : [],
    "epoch13_test_loss" : [],
    "epoch13_test_acc" : [],

    "epoch14_train_loss" : [],
    "epoch14_train_acc" : [],
    "epoch14_test_loss" : [],
    "epoch14_test_acc" : [],

    "epoch15_train_loss" : [],
    "epoch15_train_acc" : [],
    "epoch15_test_loss" : [],
    "epoch15_test_acc" : [],

    "epoch16_train_loss" : [],
    "epoch16_train_acc" : [],
    "epoch16_test_loss" : [],
    "epoch16_test_acc" : [],

    "epoch17_train_loss" : [],
    "epoch17_train_acc" : [],
    "epoch17_test_loss" : [],
    "epoch17_test_acc" : [],

    "epoch18_train_loss" : [],
    "epoch18_train_acc" : [],
    "epoch18_test_loss" : [],
    "epoch18_test_acc" : [],

    "epoch19_train_loss" : [],
    "epoch19_train_acc" : [],
    "epoch19_test_loss" : [],
    "epoch19_test_acc" : [],

    "epoch20_train_loss" : [],
    "epoch20_train_acc" : [],
    "epoch20_test_loss" : [],
    "epoch20_test_acc" : []
}

#############################################DEFINE SET TO BE CLASSIFIED###########################################################

for filename in files_SP:
    # read csv data
    print(filename)
    df = pd.read_csv("window_SP_short/"+filename)
    X, y, X_train, X_test, y_train, y_test, CLASSNUM, count_dict = outputMapSPDir(df)

    # initialize Datasets
    train_dataset = classifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = classifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    # weighted sampling
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)
        
    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in count_dict.values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    print(count_dict)
    print(class_weights)
    

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    # define model parameters
    EPOCHS = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 0.000005
    NUM_FEATURES = len(X.columns)
    NUM_DIMFACTOR = 10
    NUM_CLASSES = CLASSNUM


    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            sampler = weighted_sampler,
                            drop_last=True
                            )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)


    # check if gpu active
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)

    # initialize model, optimizer and loss function
    model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES, num_dimfactor=NUM_DIMFACTOR)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    index_key_list = 1
    
    key_list = []

    for key in stats_dict:
        key_list.append(key)

    stats_dict["name"].append(filename)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_acc = train_loop(train_loader, model, criterion, optimizer, t)
        stats_dict[key_list[index_key_list]].append(train_loss)
        index_key_list += 1
        stats_dict[key_list[index_key_list]].append(train_acc.item())
        index_key_list += 1
        test_loss, test_acc = test_loop(test_loader, model, criterion)
        stats_dict[key_list[index_key_list]].append(test_loss)
        index_key_list += 1
        stats_dict[key_list[index_key_list]].append(test_acc.item())
        index_key_list += 1
    print("Done!")

# append contents of stats_dict to dataframe
print(stats_dict)
stats_df = pd.DataFrame(stats_dict)
stats_df.to_csv("test.csv")