import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import scipy.io
from sklearn.metrics import precision_score, recall_score, f1_score


class PowerGridDataset(Dataset):
    def __init__(self, data, labels, time_steps):
        voltage_magnitudes = data[:, :14].reshape(-1, 14, 1)
        node_active_power = data[:, 54:68].reshape(-1, 14, 1)
        node_reactive_power = data[:, 68:82].reshape(-1, 14, 1)
        self.data = np.concatenate([voltage_magnitudes, node_active_power, node_reactive_power], axis=-1)
        self.labels = labels
        self.time_steps = time_steps

    def __len__(self):
        return len(self.data) - self.time_steps + 1

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.time_steps, :, :]  # (T, N, F)
        y = self.labels[idx + self.time_steps - 1, :]  # (N,)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class STGNN(nn.Module):
    def __init__(self, input_dim, hidden_lstm, hidden_gcn, num_nodes):
        super(STGNN, self).__init__()
        self.num_nodes = num_nodes
        self.lstm = nn.LSTM(input_dim, hidden_lstm, batch_first=True)
        self.gcn = nn.Linear(hidden_lstm, hidden_gcn)
        self.output = nn.Linear(hidden_gcn, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(14, 28)

    def forward(self, x, adj):
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3)
        lstm_out = []
        for i in range(N):
            node_seq = x[:, i, :, :]
            out, _ = self.lstm(node_seq)
            lstm_out.append(out[:, -1, :])
        lstm_out = torch.stack(lstm_out, dim=1)

        A_hat = adj + torch.eye(N).to(x.device)
        D_hat = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        A_norm = D_hat @ A_hat @ D_hat

        x_gcn = torch.einsum('ij,bjk->bik', A_norm, lstm_out)
        x_gcn = self.relu(self.gcn(x_gcn))
        x_gcn = self.output(x_gcn).squeeze(-1)
        out=self.sigmoid(self.linear1(x_gcn ))
        #print(out.shape)#torch.Size([64, 14])
        # input()

        return out



# 静态邻接矩阵
adj_static = np.load(r'adj_matrix_14.npy') # 替换为实际路径
adj_static = torch.from_numpy(adj_static).float()

# 初始化模型
model = STGNN(input_dim=3, hidden_lstm=2, hidden_gcn=6, num_nodes=14)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

