"""
Training for LSTM future prediction models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from util import *
from adding_features import get_np_data
import time

class TimeSeriesData(Dataset):
    def __init__(self, X, y):
        self.y = y.astype(np.float32)
        self.X = X.astype(np.float32)
        #X is (N datapoints, seq_length, num_features)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return X, y
    
class LSTM(nn.Module):
    def __init__(self, num_features = 5, hidden_size = 30, num_layers = 1, dropout_rate = 0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_block = nn.LSTM(num_features, hidden_size, num_layers, batch_first = True)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.fc_final = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        lstm_out, _ = self.lstm_block(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)

        final_out = self.fc_final(lstm_out)

        return final_out #pre non linearity?

    
def get_data_loaders(train_file, test_file, scaling, features, time_forward,  batch_size = 30):

    X_train, y_train = get_np_data(train_file, trans_back_window = 50, time_forward = time_forward, 
                step = 10, scaling = scaling, features = features)
    X_test, y_test = get_np_data(test_file, trans_back_window = 50, time_forward = time_forward, 
                step = 10, scaling = scaling, features = features)
    
    y_mean = y_train.mean()

    print(f'The size of our X train dataset is {X_train.shape}')
    print(f'The size of our X test dataset is {X_test.shape}')
    num_features = X_train.shape[2]
    print(f"Our max value is {max(y_train)}")
    print(f"Our minimum value is {min(y_train)}")

    train_data = TimeSeriesData(X_train, y_train)
    test_data = TimeSeriesData(X_test, y_test)

    #used for batching data, import to shuffle since our data is time series
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True) 
    val_data_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)  

    return train_data_loader, val_data_loader, num_features, y_mean


def train_model(num_epochs, hidden_size, train_file, test_file, scaling, features, time_forward, batch_size = 30, lr = 0.00001, dropout = 0):
    train_data_loader, test_data_loader, num_features, y_mean = get_data_loaders(train_file, test_file, scaling, features, time_forward, batch_size = batch_size)

    model = LSTM(hidden_size= hidden_size, dropout_rate= dropout, num_features= num_features)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    model.train()
    
    for epoch in range(num_epochs):
        print(epoch)
        if epoch % 10 == 0:
            evaluate_model(model, test_data_loader, y_mean)
            model.train()
        for batch_X, batch_y in train_data_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

    evaluate_model(model, test_data_loader)

def evaluate_model(model, test_data_loader, y_mean):
    print("Beginning to evaluate model")
    model.eval()
    loss_function = nn.MSELoss()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch_X, batch_y in test_data_loader:
            outputs = model(batch_X).squeeze(-1)
            size = batch_y.shape[0]
            total_loss += size * loss_function(outputs, batch_y).item()
            count += size

        
    print(f"We have an mse of {total_loss/count}")
   
    return total_loss/count

if __name__ == "__main__":
    features = ["avg_buy_streak", "life_max_price", "recent_max_price", "price", "life_transactions"]
    scaling = {"avg_buy_streak": 1, "life_max_price": 1e6, "recent_max_price": 1e8, "price": 1e7, "life_transactions": 1e-2}
    time_forward = 5 * get_length_of_time('min')
    train_model(1000, 30, "data/small_all_incorrect_points_removed_added_features.json", 
            "data/small_all_incorrect_points_removed_added_features.json", scaling, features, time_forward, batch_size = 30, lr = 0.00001)