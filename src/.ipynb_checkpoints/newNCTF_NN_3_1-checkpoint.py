import sys
import os
# 添加模块所在的文件夹到 sys.path
folder_path = "/mnt/sda/liupei/NCTF/newCode/src/pykan/"
sys.path.append(folder_path)

import random
import numpy as np
import tensorly as tl
import scipy as sp
import torch
from matplotlib import pyplot as plt
# from sklearn.manifold import TSNE
from torch import optim, nn
from torch.utils.data import DataLoader

from tensorly import unfold, fold
import torch
import tensorly as tl
import tensorly.tenalg as tl_alg

#from pykan import *
from kan import *
#import kan 
# from tensorly.tenalg import khatri_rao

# 设置 Tensorly 的后端为 PyTorch
tl.set_backend('pytorch')

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        #print(x.size())
        y = self.avg_pool(x)
        #print(y.size())
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        #print(y.size())
        # Multi-scale information fusion
        y = self.sigmoid(y)
        #print(y.size())
        out=x * y.expand_as(x)
        #print(out.size())
        return out

# class KANLayers(nn.Module):
#     def __init__(self, input_size, dims=[], act_func=nn.SiLU, dropout=0.0, input_dropout=0.2):
#         super(KANLayers, self).__init__()
#         self.input_size = input_size
#         self.layers = nn.ModuleList()  # 用于存储网络层
#         self.num_layers = len(dims)
#         # 输入层
#         self.layers.append(FastKANLayer(self.input_size, dims[0],base_activation=act_func))
#         if input_dropout > 0:
#             self.layers.append(nn.Dropout(input_dropout))
#         # 隐藏层
#         for i in range(1, self.num_layers - 1):
#             self.layers.append(FastKANLayer(dims[i - 1], dims[i],base_activation=act_func))
#             #self.layers.append(act_func)
#             if dropout > 0:
#                 self.layers.append(nn.Dropout(dropout))
#         # 输出层
#         if len(dims)>1:
#             self.layers.append(FastKANLayer(dims[-2], dims[-1],base_activation=act_func))

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

# class KANLayers(nn.Module):
#     def __init__(self, input_size, dims=[], act_func=nn.SiLU(), dropout=0.0, input_dropout=0.2):
#         super(KANLayers, self).__init__()
#         self.input_size = input_size
#         self.layers = nn.ModuleList()  # 用于存储网络层
#         self.num_layers = len(dims)
#         # 输入层
#         self.layers.append(KANLayer(in_dim=self.input_size, out_dim=dims[0]))
#         if input_dropout > 0:
#             self.layers.append(nn.Dropout(input_dropout))
#         # 隐藏层
#         for i in range(1, self.num_layers - 1):
#             self.layers.append(KANLayer(in_dim=dims[i - 1], out_dim=dims[i]))
#             #self.layers.append(act_func)
#             if dropout > 0:
#                 self.layers.append(nn.Dropout(dropout))
#         # 输出层
#         if len(dims)>1:
#             self.layers.append(KANLayer(in_dim=dims[-2], out_dim=dims[-1]))

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

class KANLayers(nn.Module):
    def __init__(self, input_size, dims=[], act_func=nn.SiLU(), dropout=0.0, input_dropout=0.2):
        super(KANLayers, self).__init__()
        self.input_size = input_size
        self.layers = nn.ModuleList()  # 用于存储网络层
        self.num_layers = len(dims)
        # 输入层
        self.layers.append(KAN(width=dims))
        # if input_dropout > 0:
        #     self.layers.append(nn.Dropout(input_dropout))
        # # 隐藏层
        # for i in range(1, self.num_layers - 1):
        #     self.layers.append(KANLayer(in_dim=dims[i - 1], out_dim=dims[i]))
        #     #self.layers.append(act_func)
        #     if dropout > 0:
        #         self.layers.append(nn.Dropout(dropout))
        # # 输出层
        # if len(dims)>1:
        #     self.layers.append(KANLayer(in_dim=dims[-2], out_dim=dims[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLPlayers(nn.Module):
    def __init__(self, input_size, dims=[], act_func=nn.GELU(), dropout= 0.0, input_dropout=0.2):
        super(MLPlayers, self).__init__()
        self.input_size = input_size
        self.layers = nn.ModuleList()  # 用于存储网络层
        # 输入层
        self.layers.append(nn.Linear(self.input_size, dims[0]))
        self.layers.append(act_func)
        if input_dropout > 0:
            self.layers.append(nn.Dropout(input_dropout))
        # 隐藏层
        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            self.layers.append(act_func)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        # 输出层
        if len(dims)>1:
            self.layers.append(nn.Linear(dims[-2], dims[-1]))
            self.layers.append(act_func)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NCTF_ConvKAN_16(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_16, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        self.eca1 = eca_layer(2, 2) ## 加权两种不同通道的特征
        self.eca2 = eca_layer(self.channel[0], rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        #self.act = nn.LeakyReLU()
        #self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)
        self.output = KAN(width=dims)

    def forward(self, x):
        ###生成初始化的index embeddings
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 将 NCTF 中 学习所得的 秩一因子矩阵 M C D 作为第二种特征输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings1 = [embeddings1[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings1[0].size())
        embeddings2 = [embeddings2[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings2[0].size())
        concatenated1 = torch.cat(embeddings1,dim=1)
        #print(concatenated1.size())
        concatenated2 = torch.cat(embeddings2,dim=1)
        #print(concatenated2.size())
        concatenated1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated1.size())
        concatenated2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated2.size())
        out= torch.cat([concatenated1,concatenated2],dim=1) ## 将两种不同特征放于两个不同通道，形成两通道数据
        #print(out.size())
        
        ### 两种不同特征位于两个不同通道，利用eca对其进行注意力加权
        out = self.eca1(out)#两种view的特征计算权重 (2,2)
        out = self.act(out)
        
        ### 卷积层
        out = self.conv1(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        out = self.conv2(out)
        out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out)
        # print(out.size())
        return out

class NCTF_ConvKAN_16_noeca1(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_16_noeca1, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        #self.eca1 = eca_layer(2, 2) ## 加权两种不同通道的特征
        self.eca2 = eca_layer(self.channel[0], rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的index embeddings
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 将 NCTF 中 学习所得的 秩一因子矩阵 M C D 作为第二种特征输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings1 = [embeddings1[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings1[0].size())
        embeddings2 = [embeddings2[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings2[0].size())
        concatenated1 = torch.cat(embeddings1,dim=1)
        #print(concatenated1.size())
        concatenated2 = torch.cat(embeddings2,dim=1)
        #print(concatenated2.size())
        concatenated1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated1.size())
        concatenated2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated2.size())
        out= torch.cat([concatenated1,concatenated2],dim=1) ## 将两种不同特征放于两个不同通道，形成两通道数据
        #print(out.size())
        
        ### 两种不同特征位于两个不同通道，利用eca对其进行注意力加权
        # out = self.eca1(out)#两种view的特征计算权重 (2,2)
        # out = self.act(out)
        
        ### 卷积层
        out = self.conv1(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        out = self.conv2(out)
        out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out.size())
        return out

class NCTF_ConvKAN_16_noeca2(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_16_noeca2, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        self.eca1 = eca_layer(2, 2) ## 加权两种不同通道的特征
        #self.eca2 = eca_layer(self.channel[0], rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的index embeddings
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 将 NCTF 中 学习所得的 秩一因子矩阵 M C D 作为第二种特征输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings1 = [embeddings1[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings1[0].size())
        embeddings2 = [embeddings2[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings2[0].size())
        concatenated1 = torch.cat(embeddings1,dim=1)
        #print(concatenated1.size())
        concatenated2 = torch.cat(embeddings2,dim=1)
        #print(concatenated2.size())
        concatenated1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated1.size())
        concatenated2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated2.size())
        out= torch.cat([concatenated1,concatenated2],dim=1) ## 将两种不同特征放于两个不同通道，形成两通道数据
        #print(out.size())
        
        ### 两种不同特征位于两个不同通道，利用eca对其进行注意力加权
        out = self.eca1(out)#两种view的特征计算权重 (2,2)
        out = self.act(out)
        
        ### 卷积层
        out = self.conv1(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        out = self.conv2(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out.size())
        return out

class NCTF_ConvKAN_16_noeca12(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_16_noeca12, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        #self.eca1 = eca_layer(2, 2) ## 加权两种不同通道的特征
        #self.eca2 = eca_layer(self.channel[0], rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的index embeddings
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 将 NCTF 中 学习所得的 秩一因子矩阵 M C D 作为第二种特征输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings1 = [embeddings1[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings1[0].size())
        embeddings2 = [embeddings2[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings2[0].size())
        concatenated1 = torch.cat(embeddings1,dim=1)
        #print(concatenated1.size())
        concatenated2 = torch.cat(embeddings2,dim=1)
        #print(concatenated2.size())
        concatenated1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated1.size())
        concatenated2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated2.size())
        out= torch.cat([concatenated1,concatenated2],dim=1) ## 将两种不同特征放于两个不同通道，形成两通道数据
        #print(out.size())
        
        ### 两种不同特征位于两个不同通道，利用eca对其进行注意力加权
        # out = self.eca1(out)#两种view的特征计算权重 (2,2)
        # out = self.act(out)
        
        ### 卷积层
        out = self.conv1(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        out = self.conv2(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out.size())
        return out

class NCTF_ConvMLP_16_noeca12(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.ReLU(), dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvMLP_16_noeca12, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        #self.eca1 = eca_layer(2, 2) ## 加权两种不同通道的特征
        #self.eca2 = eca_layer(self.channel[0], rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        self.output = MLPlayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的index embeddings
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 将 NCTF 中 学习所得的 秩一因子矩阵 M C D 作为第二种特征输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings1 = [embeddings1[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings1[0].size())
        embeddings2 = [embeddings2[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings2[0].size())
        concatenated1 = torch.cat(embeddings1,dim=1)
        #print(concatenated1.size())
        concatenated2 = torch.cat(embeddings2,dim=1)
        #print(concatenated2.size())
        concatenated1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated1.size())
        concatenated2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated2.size())
        out= torch.cat([concatenated1,concatenated2],dim=1) ## 将两种不同特征放于两个不同通道，形成两通道数据
        #print(out.size())
        
        ### 两种不同特征位于两个不同通道，利用eca对其进行注意力加权
        # out = self.eca1(out)#两种view的特征计算权重 (2,2)
        # out = self.act(out)
        
        ### 卷积层
        out = self.conv1(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        out = self.conv2(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out.size())
        return out

class NCTF_ConvMLP_16_noeca2(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.ReLU(), dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvMLP_16_noeca2, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        self.eca1 = eca_layer(2, 2) ## 加权两种不同通道的特征
        #self.eca2 = eca_layer(self.channel[0], rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        self.output = MLPlayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的index embeddings
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 将 NCTF 中 学习所得的 秩一因子矩阵 M C D 作为第二种特征输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings1 = [embeddings1[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings1[0].size())
        embeddings2 = [embeddings2[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings2[0].size())
        concatenated1 = torch.cat(embeddings1,dim=1)
        #print(concatenated1.size())
        concatenated2 = torch.cat(embeddings2,dim=1)
        #print(concatenated2.size())
        concatenated1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated1.size())
        concatenated2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated2.size())
        out= torch.cat([concatenated1,concatenated2],dim=1) ## 将两种不同特征放于两个不同通道，形成两通道数据
        #print(out.size())
        
        ### 两种不同特征位于两个不同通道，利用eca对其进行注意力加权
        out = self.eca1(out)#两种view的特征计算权重 (2,2)
        out = self.act(out)
        
        ### 卷积层
        out = self.conv1(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        out = self.conv2(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out.size())
        return out

class NCTF_ConvMLP_16(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.ReLU(), dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvMLP_16, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        self.eca1 = eca_layer(2, 2) ## 加权两种不同通道的特征
        self.eca2 = eca_layer(self.channel[0], rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        self.output = MLPlayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的index embeddings
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 将 NCTF 中 学习所得的 秩一因子矩阵 M C D 作为第二种特征输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings1 = [embeddings1[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings1[0].size())
        embeddings2 = [embeddings2[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings2[0].size())
        concatenated1 = torch.cat(embeddings1,dim=1)
        #print(concatenated1.size())
        concatenated2 = torch.cat(embeddings2,dim=1)
        #print(concatenated2.size())
        concatenated1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated1.size())
        concatenated2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated2.size())
        out= torch.cat([concatenated1,concatenated2],dim=1) ## 将两种不同特征放于两个不同通道，形成两通道数据
        #print(out.size())
        
        ### 两种不同特征位于两个不同通道，利用eca对其进行注意力加权
        out = self.eca1(out)#两种view的特征计算权重 (2,2)
        out = self.act(out)
        
        ### 卷积层
        out = self.conv1(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        out = self.conv2(out)
        out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out.size())
        return out

class NCTF_ConvKAN_13(nn.Module):
    def __init__(self,shape, rank, nc, M, C, D, device, kernel_size=[], dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_13, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc
        self.kernel_size = kernel_size
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.weight = [M, C, D]
        self.featureFusion = nn.ModuleList([nn.Linear(rank*2,rank) for i in range(len(shape))])  # 用于特征融层
        #print(kernel_size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的秩一因子矩阵 M C D
        ###这里将 NCTF 中 学习所得的 M C D 作为权重引入
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 这里将 NCTF 中 学习所得的 M C D 作为输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings12 = [torch.cat([embeddings1[i],embeddings2[i]],dim=1) for i in range(len(x))]
        #print(embeddings12[0].size())
        embeddings = [(self.featureFusion[i](embeddings12[i])) for i in range(len(x))]
        embeddings = [(self.act(embeddings[i])) for i in range(len(x))]
        #print(embeddings[0].size())
        embeddings = [embeddings[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings[0].size())
        concatenated = torch.cat(embeddings,dim=1)
        #print(concatenated.size())
        out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        #print(out.size())
        ### 卷积层
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        ### 铺平展开
        out = self.flatten(out)
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(output.size())
        return out

class NCTF_ConvKAN_1(nn.Module):
    def __init__(self,shape, rank, nc, M, C, D, device, kernel_size=[], dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_1, self).__init__()
        self.device = device
        self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc
        self.kernel_size = kernel_size
        #self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

        # self.fc1 = nn.Linear(nc, nc)
        # self.fc2 = nn.Linear(nc, 1)
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的秩一因子矩阵 M C D
        ## 这里将 NCTF 中 学习所得的 M C D 作为输入
        embeddings = [self.weight[i][x[i]] for i in range(len(x))]
        concatenated = torch.cat(embeddings, dim=1)
        # print(concatenated.size())
        out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        ### 卷积层
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        ### 铺平展开
        out = self.flatten(out)
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(output.size())
        return out

class ConvKAN(nn.Module):
    def __init__(self, shape, rank, nc,  device, kernel_size=[],dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(ConvKAN, self).__init__()
        self.device = device
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc
        self.kernel_size = kernel_size
        #print(kernel_size)
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ## 只用index embeddings
        embeddings = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        # print(len(embeddings))
        # print(embeddings[0].size(),embeddings[1].size(),embeddings[2].size())
        concatenated = torch.cat(embeddings, dim=1)
        out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        ### 卷积层
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        ### 铺平展开
        out = self.flatten(out)
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(output.size())
        return out

class Costco(nn.Module):
    def __init__(self, shape, rank, nc, device):
        super(Costco, self).__init__()
        self.device = device
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.rank=rank
        self.shape=shape
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=(rank, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(1,len(shape)), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.last_size = nc

    def forward(self,x):
        embeddings = [self.embeds[i](x[i]).view(-1,1,self.rank) for i in range(len(x))]#利用nn.Embedding生成初始化的M C D
        concatenated = torch.cat(embeddings,dim=1)
        reshaped = concatenated.view((concatenated.size(0),1,self.rank, len(self.shape)))
        #print(reshaped.size())
        conv1_out = torch.relu(self.conv1(reshaped))
        #print(conv1_out.size())
        conv2_out = torch.relu(self.conv2(conv1_out))
        #print(conv2_out.size())
        flattened = self.flatten(conv2_out)
        #print(flattened.size())
        fc1_out = torch.relu(self.fc1(flattened))
        self.intermediate = fc1_out
        #print(fc1_out.size())
        output = torch.relu(self.fc2(fc1_out))
        #print(output.size())
        return output

class CTF_DDI(nn.Module):### CTF 中的
    def __init__(self, shape, rank, hids_size=[256, 256, 128], embeds=[], device='cuda'):
        super(CTF_DDI, self).__init__()
        self.device=device
        #self.embeds = [embeds[i].to(device) for i in range(len(shape))]
        self.embeds = embeds
        self.rank = rank
        self.input_size = rank * len(shape)
        self.hids_size = hids_size
        self.Modelist = nn.ModuleList()
        #self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        input_layer = nn.Linear(self.input_size, hids_size[0])
        output_layer = nn.Linear(hids_size[-1], 1)
        self.Modelist.append(input_layer)
        self.Modelist.append(nn.ReLU())
        for i in range(len(hids_size)-1):
            self.Modelist.append(nn.Linear(hids_size[i], hids_size[i+1]))
            self.Modelist.append(nn.ReLU())
        self.Modelist.append(output_layer)

    def forward(self, x):
        embeddings = [self.embeds[i][x[i]] for i in range(len(x))]
        #print(embeddings[0].shape)
        x = torch.cat(embeddings, 1)
        #print(x.size())
        for model in self.Modelist:
            x = model(x)
            #print(x.size())
        return x

class DeepSynergy(nn.Module):
    def __init__(self, shape, rank, input_size, embeds=[], act_func=nn.ReLU(), dropout=0.5, input_dropout=0.2,
                 dims=[8182, 4096, 1]):
        super(DeepSynergy, self).__init__()
        # self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.embeds = embeds
        self.layers = nn.ModuleList()  # 用于存储网络层
        self.flatten = nn.Flatten()
        self.rank = rank
        # self.input_size = rank * len(shape)
        self.input_size = input_size
        # self.input_size = sum(shape)
        self.dims = dims

        # 输入层
        self.layers.append(nn.Linear(self.input_size, dims[0]))
        self.layers.append(nn.Dropout(input_dropout))

        # 隐藏层
        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            self.layers.append(act_func)
            self.layers.append(nn.Dropout(dropout))

        # 输出层
        if len(dims) > 1:
            self.layers.append(nn.Linear(dims[-2], dims[-1]))

    def forward(self, x):
        # embeddings = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        embeddings = [self.embeds[i][x[i]] for i in range(len(x))]
        concatenated = torch.cat(embeddings, dim=1)
        out = self.flatten(concatenated)
        # print(out.size())
        for layer in self.layers:
            out = layer(out)
            # print(out.size())
        return out

class DeepSynergy_new(nn.Module):
    def __init__(self, shape, rank, input_size, X_tr,X_val,X_train,X_test, act_func=nn.ReLU(), dropout=0.5, input_dropout=0.2,
                 dims=[8182, 4096, 1]):
        super(DeepSynergy_new, self).__init__()
        # self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        #self.embeds = embeds
        self.layers = nn.ModuleList()  # 用于存储网络层
        self.flatten = nn.Flatten()
        self.rank = rank
        # self.input_size = rank * len(shape)
        self.input_size = input_size
        # self.input_size = sum(shape)
        self.dims = dims
        self.Xtr=X_tr
        self.Xval=X_val
        self.Xtrain=X_train
        self.Xtest =X_test

        # 输入层
        self.layers.append(nn.Linear(self.input_size, dims[0]))
        self.layers.append(nn.Dropout(input_dropout))

        # 隐藏层
        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            self.layers.append(act_func)
            self.layers.append(nn.Dropout(dropout))

        # 输出层
        if len(dims) > 1:
            self.layers.append(nn.Linear(dims[-2], dims[-1]))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
            # print(out.size())
        return X

class DTF(nn.Module):
    def __init__(self, shape, rank, input_size, embeds=[], nn_struc=[2048, 1024, 512], input_dp=0.2, first_dp=0.5,
                 second_dp=0.5):
        super(DTF, self).__init__()
        self.embeds = embeds
        # self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.layers = nn.ModuleList()  # 用于存储网络层
        self.flatten = nn.Flatten()
        self.rank = rank
        # self.input_size = rank * len(shape)
        self.input_size = input_size
        # self.input_size = sum(shape)

        # 输入层
        self.layers.append(nn.Linear(self.input_size, nn_struc[0]))
        self.layers.append(nn.ReLU())
        if input_dp != 0:
            self.layers.append(nn.Dropout(input_dp))

        # 第一隐藏层
        self.layers.append(nn.Linear(nn_struc[0], nn_struc[1]))
        self.layers.append(nn.ReLU())
        if first_dp != 0:
            self.layers.append(nn.Dropout(first_dp))

        # 第二隐藏层（如果有）
        if len(nn_struc) == 3:
            self.layers.append(nn.Linear(nn_struc[1], nn_struc[2]))
            self.layers.append(nn.ReLU())
            if second_dp != 0:
                self.layers.append(nn.Dropout(second_dp))

        # 输出层
        self.layers.append(nn.Linear(nn_struc[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        # embeddings = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        embeddings = [self.embeds[i][x[i]] for i in range(len(x))]
        concatenated = torch.cat(embeddings, dim=1)
        out = self.flatten(concatenated)
        for layer in self.layers:
            out = layer(out)
        return out

class DTF_new(nn.Module):
    def __init__(self, shape, rank, input_size, embeds=[], nn_struc=[2048, 1024, 512], input_dp=0.2, first_dp=0.5,
                 second_dp=0.5):
        super(DTF_new, self).__init__()
        self.embeds = embeds
        # self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.layers = nn.ModuleList()  # 用于存储网络层
        self.flatten = nn.Flatten()
        self.rank = rank
        # self.input_size = rank * len(shape)
        self.input_size = input_size
        # self.input_size = sum(shape)

        # 输入层
        self.layers.append(nn.Linear(self.input_size, nn_struc[0]))
        self.layers.append(nn.ReLU())
        if input_dp != 0:
            self.layers.append(nn.Dropout(input_dp))

        # 第一隐藏层
        self.layers.append(nn.Linear(nn_struc[0], nn_struc[1]))
        self.layers.append(nn.ReLU())
        if first_dp != 0:
            self.layers.append(nn.Dropout(first_dp))

        # 第二隐藏层（如果有）
        if len(nn_struc) == 3:
            self.layers.append(nn.Linear(nn_struc[1], nn_struc[2]))
            self.layers.append(nn.ReLU())
            if second_dp != 0:
                self.layers.append(nn.Dropout(second_dp))

        # 输出层
        self.layers.append(nn.Linear(nn_struc[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        # embeddings = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        # embeddings = [self.embeds[i][x[i]] for i in range(len(x))]
        # concatenated = torch.cat(embeddings, dim=1)
        # out = self.flatten(concatenated)
        for layer in self.layers:
            x = layer(x)
        return x

class NCTF_ConvKAN_17(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_17, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.weight = [M, C, D]
        self.featureFusion = nn.ModuleList([nn.Linear(rank*2,rank) for i in range(len(shape))])  # 用于特征融层
        #print(kernel_size)
        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size[0], padding=0)
        # self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
        # self.conv3 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[2], padding=0)
        #self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size[0], padding=0)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        #self.eca1 = eca_layer(2, 2)
        self.eca2 = eca_layer(self.channel[0], rank)
        #self.conv3 = nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=kernel_size[0], padding=0)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

        # self.fc1 = nn.Linear(nc, nc)
        # self.fc2 = nn.Linear(nc, 1)
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 这里将 NCTF 中 学习所得的 M C D 作为输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings12 = [torch.cat([embeddings1[i],embeddings2[i]],dim=1) for i in range(len(x))]
        #print(embeddings12[0].size())
        embeddings = [(self.featureFusion[i](embeddings12[i])) for i in range(len(x))]
        embeddings = [(self.act(embeddings[i])) for i in range(len(x))]
        #print(embeddings[0].size())
        embeddings = [embeddings[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings[0].size())
        concatenated = torch.cat(embeddings,dim=1)
        #print(concatenated.size())
        out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape))) ## 1通道
        
        ### 卷积层
        # out = self.eca1(out)#两种view的特征计算权重
        # out = self.act(out)
        
        out = self.conv1(out)
        #out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        
        out = self.conv2(out)
        out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        #print(out.size())
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out.size())
        return out

class NCTF_ConvKAN_11(nn.Module):
    def __init__(self,shape, rank, nc, M, C, D, device, kernel_size=[], dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_11, self).__init__()
        self.device = device
        self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc
        self.kernel_size = kernel_size
        #self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
        self.eca2 = eca_layer(self.channel, rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

        # self.fc1 = nn.Linear(nc, nc)
        # self.fc2 = nn.Linear(nc, 1)
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的秩一因子矩阵 M C D
        ## 这里将 NCTF 中 学习所得的 M C D 作为输入
        embeddings = [self.weight[i][x[i]] for i in range(len(x))]
        concatenated = torch.cat(embeddings, dim=1)
        # print(concatenated.size())
        out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        ### 卷积层
        out = self.conv1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        ### 铺平展开
        out = self.flatten(out)
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(output.size())
        return out

class ConvKAN_1(nn.Module):
    def __init__(self, shape, rank, nc,  device, kernel_size=[],dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(ConvKAN_1, self).__init__()
        self.device = device
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc
        self.kernel_size = kernel_size
        #print(kernel_size)
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
        self.eca2 = eca_layer(self.channel, rank)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ## 只用index embeddings
        embeddings = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        # print(len(embeddings))
        # print(embeddings[0].size(),embeddings[1].size(),embeddings[2].size())
        concatenated = torch.cat(embeddings, dim=1)
        out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        ### 卷积层
        out = self.conv1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.eca2(out)#2r个特征计算权重
        out = self.act(out)
        ### 铺平展开
        out = self.flatten(out)
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(output.size())
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class NCTF_ConvKAN_CBAM(nn.Module):
    def __init__(self,shape, rank, M, C, D, device, kernel_size=[], nc=[2,2],dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_CBAM, self).__init__()
        self.device = device
        #self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc[-1]
        self.kernel_size = kernel_size
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.channel[0], kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel[0], out_channels=self.channel[1], kernel_size=kernel_size[1], padding=0)
        #self.eca1 = eca_layer(2, 2) ## 加权两种不同通道的特征
        #self.eca2 = eca_layer(self.channel[0], rank)
        self.eca1 = CBAM(2,2) 
        self.eca2 = CBAM(self.channel[0]) 
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        #self.act = nn.LeakyReLU()
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ###生成初始化的index embeddings
        embeddings1 = [(self.embeds[i](x[i])) for i in range(len(x))]
        #print(embeddings1[0].size())
        ## 将 NCTF 中 学习所得的 秩一因子矩阵 M C D 作为第二种特征输入
        embeddings2 = [self.weight[i][x[i]] for i in range(len(x))]
        #print(embeddings2[0].size())
        embeddings1 = [embeddings1[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings1[0].size())
        embeddings2 = [embeddings2[i].view(-1,1,self.rank) for i in range(len(x))]
        #print(embeddings2[0].size())
        concatenated1 = torch.cat(embeddings1,dim=1)
        #print(concatenated1.size())
        concatenated2 = torch.cat(embeddings2,dim=1)
        #print(concatenated2.size())
        concatenated1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated1.size())
        concatenated2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        #print(concatenated2.size())
        out= torch.cat([concatenated1,concatenated2],dim=1) ## 将两种不同特征放于两个不同通道，形成两通道数据
        #print(out.size())
        
        ### 两种不同特征位于两个不同通道，利用eca对其进行注意力加权
        out = self.eca1(out)#两种view的特征计算权重 (2,2)
        out = self.act(out)
        
        ### 卷积层
        out = self.conv1(out)
        out = self.act(out)
        out = self.eca2(out)#2r个特征计算权重
        #print(out.size())
        
        out = self.conv2(out)
        out = self.act(out)
        #out = self.eca2(out)#2r个特征计算权重
        #print(out.size())
        
        ### 铺平展开
        out = self.flatten(out)
        #print(out.size())
        
        ### 输出
        # out = torch.relu(self.fc1(out))
        out = self.output(out)
        # print(out.size())
        return out

# class NCTF_ConvKAN_18(nn.Module):
#     def __init__(self,shape, rank, nc, M, C, D, device, kernel_size=[], dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
#         super(NCTF_ConvKAN_18, self).__init__()
#         # self.device = device
#         # self.weight = [M, C, D]
#         # self.rank = rank
#         # self.shape = shape
#         # self.dims = dims
#         # self.channel = nc
#         # self.input_size = nc
#         # self.kernel_size = kernel_size
#         #self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
#         # self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[0], padding=0)
#         # self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
#         self.output1 = NCTF_ConvKAN_1(shape, rank, nc, M, C, D, device, kernel_size, dims, act_func, dropout, input_dropout)
#         self.output2 = ConvKAN(shape, rank, nc,  device, kernel_size, dims, act_func, dropout, input_dropout)
#         # self.flatten = nn.Flatten()
#         # self.act = nn.ReLU()

#         # # self.fc1 = nn.Linear(nc, nc)
#         # # self.fc2 = nn.Linear(nc, 1)
#         # self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

#     def forward(self, x):
#         ###生成初始化的秩一因子矩阵 M C D
#         ## 这里将 NCTF 中 学习所得的 M C D 作为输入
#         # embeddings = [self.weight[i][x[i]] for i in range(len(x))]
#         # concatenated = torch.cat(embeddings, dim=1)
#         # # print(concatenated.size())
#         # out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
#         # ### 卷积层
#         # out = self.conv1(out)
#         # out = self.act(out)
#         # out = self.conv2(out)
#         # out = self.act(out)
#         # ### 铺平展开
#         # out = self.flatten(out)
#         ### 输出
#         # out = torch.relu(self.fc1(out))
#         out1 = self.output1(x)
#         #print(out1.size())
#         out2 = self.output2(x)
#         #print(out1.size())
#         #print(out2.size())
#         out = (out1+out2)/2
#         #print(out.size())
#         return out

class NCTF_ConvKAN_18(nn.Module):
    def __init__(self,shape, rank, nc, M, C, D, device, kernel_size=[], dims=[], act_func=nn.SiLU, dropout=0.5, input_dropout=0.2):
        super(NCTF_ConvKAN_18, self).__init__()
        self.device = device
        self.weight = [M, C, D]
        self.rank = rank
        self.shape = shape
        self.dims = dims
        self.channel = nc
        self.input_size = nc
        self.kernel_size = kernel_size
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=kernel_size[0], padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=kernel_size[1], padding=0)
        # self.output1 = NCTF_ConvKAN_1(shape, rank, nc, M, C, D, device, kernel_size, dims, act_func, dropout, input_dropout)
        # self.output2 = ConvKAN(shape, rank, nc,  device, kernel_size, dims, act_func, dropout, input_dropout)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()

        # self.fc1 = nn.Linear(nc, nc)
        # self.fc2 = nn.Linear(nc, 1)
        self.output = KANLayers(self.input_size, self.dims, act_func, dropout, input_dropout)

    def forward(self, x):
        ##生成初始化的秩一因子矩阵 M C D
        # 这里将 NCTF 中 学习所得的 M C D 作为输入
        embeddings1 = [self.weight[i][x[i]] for i in range(len(x))]
        concatenated1 = torch.cat(embeddings1, dim=1)
        # print(concatenated.size())
        out1 = concatenated1.view((concatenated1.size(0), 1, self.rank, len(self.shape)))
        ### 卷积层
        out1 = self.conv1(out1)
        out1 = self.act(out1)
        out1 = self.conv2(out1)
        out1 = self.act(out1)
        ### 铺平展开
        out1 = self.flatten(out1)
        ### 输出
        out1 = self.output(out1)

        ## 只用index embeddings
        embeddings2 = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        # print(len(embeddings))
        # print(embeddings[0].size(),embeddings[1].size(),embeddings[2].size())
        concatenated2 = torch.cat(embeddings2, dim=1)
        out2 = concatenated2.view((concatenated2.size(0), 1, self.rank, len(self.shape)))
        ### 卷积层
        out2 = self.conv1(out2)
        out2 = self.act(out2)
        out2 = self.conv2(out2)
        out2 = self.act(out2)
        ### 铺平展开
        out2 = self.flatten(out2)
        ### 输出
        out2 = self.output(out2)
        
        ### 输出
        out = (out1+out2)/2  
        return out