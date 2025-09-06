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
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold



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
#104个特征
class FeatureAttentionLayer(nn.Module):
    def __init__(self, n_features,dropout, alpha, embed_dim=104, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.use_bias = use_bias

        # 使用GATv2时，直接设置嵌入维度，不需要加倍
        self.embed_dim = embed_dim

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = 1
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        #self.a = nn.Parameter(torch.empty((2 * self.embed_dim, 1)))  # 用于计算注意力系数的参数
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, x):
        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)
            # (b, k, k, 2)

            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)
            # print(e.shape)
            # input()
        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # 假设 attention 的形状已经是 [32, 104, 104]
        # x 的形状是 [32, 104]

        # 临时调整 x 的形状以便进行矩阵乘法
        x_expanded = x.unsqueeze(-1)  # x_expanded 的形状变为 [32, 104, 1]

        # 执行批量矩阵乘法
        h = torch.bmm(attention, x_expanded)  # h 的形状将是 [32, 104, 1]

        # 将 h 压缩回原始的 x 形状
        h_squeezed = h.squeeze(-1)  # h_squeezed 的形状是 [32, 104]

        # 应用激活函数
        h_final = self.sigmoid(h_squeezed)

        return h_final
    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        """

        K = v.shape[1]  # Number of features, 104 in your case
        batch_size = v.shape[0]

        # Expand v to (batch_size, n_features, 1) to prepare for repeating
        v_expanded = v.unsqueeze(-1)  # Shape: [32, 104, 1]

        # Repeat features along the new dimension to simulate blocks_repeating
        # We want each feature to repeat K times consecutively
        blocks_repeating = v_expanded.repeat(1, 1, K)  # Shape: [32, 104, 104]

        # For blocks_alternating, we need each feature to be repeated K times in blocks
        blocks_alternating = v_expanded.repeat(1, K, 1)  # Shape: [32, 104*104, 1]

        # Now, we reshape blocks_repeating and blocks_alternating to match and concatenate along the last dimension
        blocks_repeating = blocks_repeating.view(batch_size, K * K, 1)  # Shape: [32, 104*104, 1]
        # blocks_alternating already has the correct shape

        # Concatenate along the feature dimension
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # Shape: [32, 104*104, 2]


        if self.use_gatv2:
            # print(combined.shape)
            # input()
            return combined.view(v.size(0), K, K, 2 )

        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class MultiHeadGATv2(nn.Module):
    def __init__(self,n_features, n_heads, dropout, alpha, concat=False, embed_dim=104, use_bias=True ):
        super(MultiHeadGATv2, self).__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.use_bias = use_bias
        self.dropout = nn.Dropout(dropout)


        # 初始化每个头的GATv2层
        self.heads = nn.ModuleList([
            FeatureAttentionLayer(n_features,dropout, alpha, embed_dim, use_bias) for _ in range(n_heads)
        ])

    def forward(self, x):
        # x的形状为[batch_size, n_features], 直接处理无需调整形状
        head_outputs = [head(x) for head in self.heads]  # 并行处理多头

        if self.concat:
            # 拼接多头的输出
            x = torch.cat(head_outputs, dim=-1)  # 沿特征维度拼接
        else:
            # 平均多头的输出
            x = torch.stack(head_outputs, dim=0).mean(dim=0)

        return x




# 修改MultiLabelClassificationModel以整合新的残差块和特征融合层
class MultiLabelClassificationModel(nn.Module):
    def __init__(self, n_features, n_classes,n_heads=3, dropout=0.3, alpha=0.2, embed_dim=104, use_gatv2=True, use_bias=True,channels=1, output_channels=32, large_kernel_size=9,medium_kernel_size=5, small_kernel_size=3):
        super(MultiLabelClassificationModel, self).__init__()
        # 使用多头 GATv2
        self.feature_attention = MultiHeadGATv2(
            n_features, n_heads, dropout=0.3, alpha=0.25, concat=False, embed_dim=embed_dim, use_bias=use_bias
        )

        # 并行一维卷积网络

        # 并行一维卷积网络
        self.fc1 = nn.Linear(n_features, n_features*64)
        self.fc2 = nn.Linear(n_features*64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 应用多头图注意力

        attention_features1 = self.feature_attention(x)
        # print(attention_features.shape)
        # input()

        # attention_features = x.unsqueeze(1)
        # attention_features1=attention_features1.unsqueeze(1)

        # 分类
        x = self.fc1(attention_features1)
        x2 = self.fc2(x)
        output=self.sigmoid(x2)

        return output
# 设置种子
set_seed(42)
