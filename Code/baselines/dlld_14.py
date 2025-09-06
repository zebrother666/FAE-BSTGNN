import time

import scipy.io
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import os
import pandas as pd



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 1. 确定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


class DLLD(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DLLD, self).__init__()

        # 根据提供的表格定义层
        self.layers = nn.Sequential(
            # 第1层：卷积层
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            # 第2层：卷积层
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            # 第3层：卷积层
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            # 第4层：卷积层
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )

        # Flatten层在PyTorch中不需要定义，因为可以在前向传播中处理

        # 全连接层
        self.fc = nn.Linear(input_channels * 64, num_classes)

        # 输出层使用Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入通过卷积层
        x = self.layers(x)

        # 为全连接层将输出扁平化
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)

        # 应用Sigmoid激活函数得到最终预测结果
        x = self.sigmoid(x)

        return x

# 设置种子
set_seed(42)
model =DLLD(input_channels=features.shape[2], num_classes=labels.shape[1]).to(device)
