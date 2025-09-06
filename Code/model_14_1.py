import scipy.io
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from src.mtcl import graph_constructor
import torch.nn.functional as F
from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
import torch.fft
from pytorch_wavelets import DWT2D, IDWT2D
import torch
import numpy as np
import pandas as pd



class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        # 修改卷积层以适应输入维度
        self.query_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim * 3, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim * 3, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim * 3, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax1 = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [B, C, 3, H, W]
        B, C, D, H, W = x.size()#torch.Size([32, 32, 3, 57, 11])
        #print(x.shape)
        # 重塑输入以适应 2D 卷积
        x_reshaped = x.view(B, C * D, H, W)

        # 应用卷积
        proj_query = self.query_conv(x_reshaped).view(B, -1, H * W).permute(0, 2, 1)  # B X (H*W) X C
        proj_key = self.key_conv(x_reshaped).view(B, -1, H * W)  # B X C x (H*W)
        energy = torch.bmm(proj_query, proj_key)  # B X (H*W) X (H*W)
        attention = self.softmax1(energy)  # B X (H*W) X (H*W) #torch.Size([32, 627, 627])
       


        proj_value = self.value_conv(x_reshaped).view(B, -1, H * W)  # B X C X (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, D, H, W)

        # out = self.gamma * out + x
        #input()
        return out


# 小波分解模块
class WaveConv2d(nn.Module):
    def __init__(self, embed_dim):
        super(WaveConv2d, self).__init__()
        # self.dummy = x
        self.in_channels = embed_dim
        self.out_channels = embed_dim
        self.level = 5
        # self.wave="sym9"
        self.wave = "db6"
        self.mode = "zero"

        self.dwt2d = DWT2D(wave=self.wave, J=self.level, mode=self.mode)
        self.idwt2d = IDWT2D(wave=self.wave, mode=self.mode)

        self.sa_c = Self_Attn(in_dim=embed_dim)

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(3 * embed_dim, 3 * embed_dim, kernel_size=1)
        self.gelu1 = nn.GELU()


    def mul2d(self, input, weights):
        return torch.einsum("bchw,ochw->bohw", input, weights)

    def forward(self, x):
      
        x_ft, x_coeffs = self.dwt2d(x)
       
        out_ft = x_ft

        for i in range(len(x_coeffs)):
            # 直接将整个 x_coeffs[i] 传递给 self.sa_c
            x_coeffs[i] = self.sa_c(x_coeffs[i])
            x_coeffs[i] = self.gelu1(x_coeffs[i])
            # 重塑以适应 conv2
            B, C, D, H, W = x_coeffs[i].shape

            # x_coeffs[i]=x_coeffs[i].view(B, C , D, H, W)
            x_coeffs[i] = self.conv2(x_coeffs[i].view(B, C * D, H, W)).view(B, C, D, H, W)

       
        x = self.idwt2d((out_ft, x_coeffs))
       
        return x


# MAWNO模块
class MAWNOBlock(nn.Module):
    def __init__(self, embed_dim, dim):
        super(MAWNOBlock, self).__init__()
        self.embed_dim = embed_dim
        self.filter = WaveConv2d(self.embed_dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=(1, 1))
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x1 = self.filter(x)
        x2 = self.conv(x)
        # x2=x
        x = x1 + x2
        x = self.gelu2(x)
        return x


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


# 门控tcn尝试

class GatedTCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(GatedTCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp2d(padding[1])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp2d(padding[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # self.linear = nn.Linear(423,419)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        hidden, gate = self.conv1(x).chunk(2, dim=1)
        hidden = hidden * self.sigmoid1(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp1(hidden)
        hidden = self.dropout1(hidden)
        hidden, gate = self.conv2(hidden).chunk(2, dim=1)
        hidden = hidden * self.sigmoid2(gate)
        hidden = torch.cat((hidden, gate), dim=1)
        hidden = self.chomp2(hidden)
        hidden = self.dropout2(hidden)
        res = x if self.downsample is None else self.downsample(x)
        return hidden + res


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1, dropout=0.2):
        super(GatedResidualBlock, self).__init__()
        padding = (kernel_size - 1) * dilation  # 计算需要的填充以保持长度不变

        self.conv_gate = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                                               stride=(1, stride), padding=(0, padding), dilation=(1, dilation)))
        self.conv_trans = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                                                stride=(1, stride), padding=(0, padding), dilation=(1, dilation)))

        self.chomp_gate = Chomp2d(padding)
        self.chomp_trans = Chomp2d(padding)
        self.sigmoid3 = nn.Sigmoid()
        self.tanh1 = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=(1, 1)) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x
        gate1 = self.conv_gate(x)
        gate2 = self.chomp_gate(gate1)
        gate = self.sigmoid3(gate2)
        # print(gate1.shape)
        # print(gate2.shape)
        trans1 = self.conv_trans(x)
        trans2 = self.chomp_trans(trans1)
        trans = self.tanh1(trans2)
        # print(trans1.shape)
        # print(trans2.shape)
        # input()

        x = self.dropout(gate * trans)

        # return x + residual
        return x


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        # adj normalization
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        # graph propagation
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)  # alpha=0.05
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class GTCNGCN(nn.Module):
    def __init__(self, num_nodes, gcn_output_size, k, node_dim, device, tcn_hidden_size, tcn_output_size, kernel_size=3,
                 gcn_depth=2, dropout=0.3, propalpha=0.4, tanhalpha=3, layers_num=3, layer_norm_affline=True,
                 static_feat=None):
        super(GTCNGCN, self).__init__()
        self.device = device
        layers = []
        layers1 = []
        num_channels = [32, 64, 128]
        # num_levels = len(num_channels)
        self.gc = graph_constructor(num_nodes, k, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)
        self.idx = torch.arange(num_nodes).to(device)
        self.gconv1 = mixprop(128, 128, gcn_depth, dropout, propalpha)
        self.gconv2 = mixprop(128, 128, gcn_depth, dropout, propalpha)
        # self.seq_node = SeqAdjust(tcn_output_size, gcn_output_size)
        for i in range(layers_num):
            dilation_size = 2 ** i
            # in_channels = tcn_hidden_size
            # out_channels = tcn_output_size
            in_channels = 32 if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(GatedResidualBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                             dropout=dropout))
           
        num_channels1 = [32, 64, 128]
        for i in range(layers_num):
            dilation_size = 2 ** i
            # in_channels = tcn_hidden_size
            # out_channels = tcn_output_size
            in_channels = 32 if i == 0 else num_channels1[i - 1]
            out_channels = num_channels1[i]
            layers1.append(GatedResidualBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                              dropout=dropout))

        self.tcn = nn.Sequential(*layers)
        self.end_tcn = nn.Sequential(*layers1)
        self.linear1 = nn.Linear(64 * 104, 28)
        # self.linear2 = nn.Linear(130
        #                          , 28)
        self.sigmoid4 = nn.Sigmoid()
        self.start_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1))
        self.start_conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 1))
        self.trans_conv = nn.Conv2d(in_channels=gcn_output_size,
                                    out_channels=tcn_hidden_size,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv1 = nn.Conv2d(in_channels=128,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   bias=True)

        self.BN1 = nn.BatchNorm2d(tcn_hidden_size)
        self.BN2 = nn.BatchNorm2d(64)
        self.leakyrelu1 = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
        self.MAWNOBlock = MAWNOBlock(32, 32)

    def forward(self, x, idx=None,return_intermediate=False):
        # 添加一个维度作为通道维度
        intermediate_outputs = {}  # 用于存储中间输出
        # if return_intermediate:
        #     intermediate_outputs['original_features'] = x

        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)

        residual = self.start_conv(x)
        # if return_intermediate:
        #     intermediate_outputs['conv_feature'] = residual
        residual = self.MAWNOBlock(residual)
        # if return_intermediate:
        #     intermediate_outputs['FAE_features'] = residual
        # residual1=self.start_conv1(x)
        # print(residual.shape)
        tcn_output = self.tcn(residual)
        # if return_intermediate:
        #     intermediate_outputs['tcn_features'] =tcn_output
        # print(tcn_output.shape)
        # input()
        if idx is None:
            adp = self.gc(self.idx)
        else:
            adp = self.gc(idx)
       
        gcn_output = self.gconv1(tcn_output, adp) + self.gconv2(tcn_output, adp.transpose(1, 0))
        
        x = gcn_output
        # if return_intermediate:
        #     intermediate_outputs['BGNN'] = gcn_output
        x = self.end_conv1(x)
        x = self.BN2(x)
        x = self.leakyrelu1(x)
        x = x.permute(0, 1, 3, 2)
        x = x[:, :, -1, :]
        x = x.view(x.size(0), -1)
      
        x = self.linear1(x)
       
        x = self.sigmoid4(x)
       
        return x



