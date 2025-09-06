import time

import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,accuracy_score
from torch.utils.data import Dataset, DataLoader

import argparse
import datetime
import random
import os
import pandas as pd


class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels, seq_length=12,normalize=True):
        self.seq_length = seq_length
        self.normalize = normalize
        if normalize:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0)
            # 避免除零错误
            self.feature_std = np.where(self.feature_std == 0, 1e-7, self.feature_std)

            # 标准化特征
            self.features = (features - self.feature_mean) / self.feature_std
        else:
            self.features = features
        self.labels = labels

    def __len__(self):
        # 根据序列长度计算可以生成的样本数
        return len(self.features) - self.seq_length + 1

    def __getitem__(self, idx):
        # 生成时间序列样本和对应的标签
        x = self.features[idx:idx+self.seq_length]
        y = self.labels[idx+self.seq_length-1]  # 取序列最后一步的标签
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def inverse_normalize(self, data):
        if self.normalize:
            return data * self.feature_std + self.feature_mean

def set_seed(seed=42):
    """设置随机数种子以确保代码的可重复性."""
    random.seed(seed)  # Python内置的random模块
    np.random.seed(seed)  # Numpy库
    torch.manual_seed(seed)  # 为CPU设置随机数种子
    torch.cuda.manual_seed(seed)  # 为所有CUDA设备设置随机数种子
    torch.cuda.manual_seed_all(seed)  # 为所有CUDA设备设置随机数种子（当使用多个GPU时）
    os.environ['PYTHONHASHSEED'] = str(seed)  # 通过环境变量设置Python哈希随机化的种子

    # 以下两行是为了确保PyTorch能尽可能地实现确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义GC_LSTM模型
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x=x.squeeze(1)
        x=x.unsqueeze(-1)
        # print(adj.shape)
        # print(x.shape)
        batch_size = x.size(0)
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        # 现在adj的shape是 (batch_size, num_nodes, num_nodes) -> (32, 104, 104)

        # # 确保x是3D的
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(0)
        # GCN层计算
        # print(adj.shape)
        # print(x.shape)
        out = torch.bmm(adj, x)  # 邻接矩阵和特征的乘积
        out = self.fc(out)
        return torch.relu(out)


class GCN_LSTM(nn.Module):
    def __init__(self, node_count, in_features, gcn_hidden_size, lstm_hidden_size, num_layers, num_classes):
        super(GCN_LSTM, self).__init__()
        self.node_count = node_count
        self.gcn = GCNLayer(in_features, gcn_hidden_size)
        self.lstm = nn.LSTM(gcn_hidden_size * node_count, lstm_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 60)
        self.fc2=nn.Linear(60, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        x=x.unsqueeze(2)
        batch_size, seq_len, node_count, in_features = x.size()
        gcn_out = []

        # 通过每个时间步的GCN
        for t in range(seq_len):
            gcn_out.append(self.gcn(x[:, t, :, :], adj))

        # 合并每个时间步的输出
        gcn_out = torch.stack(gcn_out, dim=1)  # (batch, seq_len, node_count, gcn_hidden_size)
        gcn_out = gcn_out.view(batch_size, seq_len, -1)  # 展平成LSTM输入形状

        # 通过LSTM层
        lstm_out, _ = self.lstm(gcn_out)

        # 获取LSTM最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 全连接层分类
        out = self.fc(lstm_out)
        out=self.fc2(out)
        out = self.sigmoid(out)
        return out




def main():
    parser = argparse.ArgumentParser()
    # dataset configurations
    parser.add_argument("--dataset", type=str, default="fdia_14", help="Dataset name.")
    parser.add_argument("--file_path", type=str,
                        default="E:\\PycharmProjects\\last_second\\dataset\\MODEL_data.mat",
                        help="Dataset filepath.")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="Test ratio.")
    #parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--slide_win", type=int, default=12, help="Slide window size.")
    parser.add_argument("--slide_stride", type=int, default=1, help="Slide window stride.")
    # model configurations
    parser.add_argument("--batch_size", type=int, default=32
                        , help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=30
                        , help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.000005, help="Weight decay.")
    parser.add_argument("--device", type=int, default=0, help="Training cuda device, -1 for cpu.")
    parser.add_argument("--random_seed", type=int, default=42
                        , help="Random seed.")
    # config
    args = parser.parse_args()  # 使用 parser.parse_args() 解析命令行参数，将其保存在 args 对象中。

    # set random seed
    # 调用 seed_everything(args.random_seed) 来设置随机种子，以确保实验的可重复性。
    set_seed(args.random_seed)
    # 创建一个 PyTorch 设备对象 device，用于指定模型在 CPU 或 GPU 上运行。
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda:{}".format(args.device))

    # load datasets and dataloaders
    print("Loading datasets...")

# 总之，这段代码实现了一个端到端的深度学习模型训练流程，包括数据加载、模型构建、优化器设置、训练循环以及验证集的评估。模型在训练过程中会保存检查点，并在验证损失不再降低时触发早停。

if __name__ == '__main__':
    main()
#Average inference time per sample: 0.0462 ms



