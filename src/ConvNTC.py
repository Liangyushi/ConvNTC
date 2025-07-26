import random
import numpy as np
import tensorly as tl
import scipy as sp
import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader

from tensorly import unfold, fold
import torch
import tensorly as tl
import tensorly.tenalg as tl_alg

from ConvKAN.kans.layers import FastKANLayer
# from tensorly.tenalg import khatri_rao
# 设置 Tensorly 的后端为 PyTorch
tl.set_backend('pytorch')

class KANLayers(nn.Module):
    def __init__(self, input_size, dims=[], act_func=nn.SiLU, dropout=0.0, input_dropout=0.2):
        super(KANLayers, self).__init__()
        self.input_size = input_size
        self.layers = nn.ModuleList()  # 用于存储网络层
        self.num_layers = len(dims)
        # 输入层
        self.layers.append(FastKANLayer(self.input_size, dims[0],base_activation=act_func))
        if input_dropout > 0:
            self.layers.append(nn.Dropout(input_dropout))
        # 隐藏层
        for i in range(1, self.num_layers - 1):
            self.layers.append(FastKANLayer(dims[i - 1], dims[i],base_activation=act_func))
            #self.layers.append(act_func)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        # 输出层
        if len(dims)>1:
            self.layers.append(FastKANLayer(dims[-2], dims[-1],base_activation=act_func))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ConvNTC(nn.Module):
    def __init__(self,shape, rank, nc, M, C, D, device, kernel_size=[], dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2,alpha=0.5):
        super(ConvNTC, self).__init__()
        self.device = device
        self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.embeds1 = nn.Embedding(num_embeddings=shape[0], embedding_dim=rank)
        self.embeds2 = nn.Embedding(num_embeddings=shape[2], embedding_dim=rank)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ##生成初始化的秩一因子矩阵 M C D
        # 这里将 NCTF 中 学习所得的 M C D 作为输入
        embeddings1 = [self.weight[i][x[i]] for i in range(len(x))]
        concatenated1 = torch.cat(embeddings1, dim=1)
        out1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        ### 卷积层
        out1 = self.conv1(out1)
        out1 = self.act(out1)
        out1 = self.conv2(out1)
        out1 = self.act(out1)
        ### 铺平展开
        out1 = self.flatten(out1)
        ### KANpredictor输出
        out1 = self.output(out1)

        ## 只用index embeddings
        e1 = self.embeds1(x[0])
        e2 = self.embeds1(x[1])
        e3 = self.embeds2(x[2])
        embeddings2 = [e1,e2,e3]
        concatenated2 = torch.cat(embeddings2, dim=1)
        out2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        ### 卷积层
        out2 = self.conv1(out2)
        out2 = self.act(out2)
        out2 = self.conv2(out2)
        out2 = self.act(out2)
        ### 铺平展开
        out2 = self.flatten(out2)
        ### KANpredictor 输出
        out2 = self.output(out2)

        ### 最终输出 看比值
        out = self.alpha * out1 + (1-self.alpha) * out2

        return out