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

from kans.layers import KANLayer, FastKANLayer
from kans.kan import KAN,FastKAN
# from tensorly.tenalg import khatri_rao

# 设置 Tensorly 的后端为 PyTorch
# tl.set_backend('pytorch')

class NCTF_CostcoModel(nn.Module):
    def __init__(self, shape, rank, nc, M, C, D, device):
        super(NCTF_CostcoModel, self).__init__()
        # self.name = name
        self.device = device
        self.weight = [M, C, D]
        self.embeds = nn.ModuleList(
            [nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        # self.embeds = [M,C,D]
        self.rank = rank
        self.shape = shape
        # self.concat = nn.Concatenate(dim=1)
        # self.reshape = nn.Reshape((rank, len(shape), 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=(1, len(shape)), padding=0)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(rank, 1), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.last_size = nc

    def forward(self, x):
        ###生成初始化的秩一因子矩阵 M C D
        ###这里将 NCTF 中 学习所得的 M C D 作为权重引入
        embeddings = [(self.embeds[i](x[i]) * self.weight[i][x[i]]).view(-1, 1, self.rank) for i in range(len(x))]
        # embeddings = [self.embeds[i][x[i]].view(-1, 1, self.rank) for i in range(len(x))]
        # embeddings = [M,C,D]
        # embeds = [
        #     self.embeds(output_dim=rank, input_dim=shape[i])(inputs[i])
        #     for i in range(len(shape))
        # ]
        # print(len(embeddings))
        # print(embeddings[0].size(),embeddings[1].size(),embeddings[2].size())
        # print(embeddings[0],embeddings[1],embeddings[2])

        concatenated = torch.cat(embeddings, dim=1)
        # print(concatenated.size())
        # reshaped = concatenated.view((concatenated.size(0),self.rank,len(self.shape),1))
        reshaped = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        # print(reshaped.size())
        conv1_out = torch.relu(self.conv1(reshaped))
        # print(conv1_out.size())
        conv2_out = torch.relu(self.conv2(conv1_out))
        # print(conv2_out.size())
        flattened = self.flatten(conv2_out)
        # print(flattened.size())
        fc1_out = torch.relu(self.fc1(flattened))
        self.intermediate = fc1_out
        # print(fc1_out.size())
        output = torch.relu(self.fc2(fc1_out))
        # print(output.size())
        return output

class NCTF_newCostcoModel(nn.Module):
    def __init__(self, shape, rank, nc, M, C, D, device):
        super(NCTF_newCostcoModel, self).__init__()
        self.device = device
        self.embeds = nn.ModuleList(
            [nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.rank = rank
        self.shape = shape
        # self.weight = [utils.weight_norm(i.cpu().numpy()).to(device) for i in [M,C,D]]
        self.weight = [M, C, D]
        # self.concat = nn.Concatenate(dim=1)
        # self.reshape = nn.Reshape((rank, len(shape), 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=(rank, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(1, len(shape)), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.last_size = nc
        # self.fc1 = utils.weight_norm(self.fc1)  # Apply weight normalization
        self.bn = nn.BatchNorm1d(self.rank)  # Batch Normalization after FC layer

    def forward(self, x):
        ###生成初始化的秩一因子矩阵 M C D
        ###这里将 NCTF 中 学习所得的 M C D 作为权重引入
        # embeddings = [(self.embeds[i](x[i])*self.weight[i][x[i]]).view(-1,1,self.rank) for i in range(len(x))]
        # embeddings = [(self.embeds[i](x[i])self.bn(self.weight[i][x[i]])).view(-1,1,self.rank) for i in range(len(x))]
        embeddings = [(self.weight[i][x[i]]).view(-1, 1, self.rank) for i in range(len(x))]
        # embeddings = [(self.embeds[i](x[i])) for i in range(len(x))]
        # #print(embeddings[0].shape)
        # weight = [self.bn(self.weight[i][x[i]]) for i in range(len(x))]
        # #print(weight[0].shape)
        # embeds = [(embeddings[i]*weight[i]).view(-1,1,self.rank) for i in range(len(x))]
        # #print(embeds[0].shape)
        # concatenated = torch.cat(embeds,dim=1)
        concatenated = torch.cat(embeddings, dim=1)
        # print(concatenated.size())
        # reshaped = concatenated.view((concatenated.size(0),self.rank,len(self.shape),1))
        reshaped = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        # print(reshaped.size())
        conv1_out = torch.relu(self.conv1(reshaped))
        # print(conv1_out.size())
        conv2_out = torch.relu(self.conv2(conv1_out))
        # print(conv2_out.size())
        flattened = self.flatten(conv2_out)
        # print(flattened.size())
        fc1_out = torch.relu(self.fc1(flattened))
        self.intermediate = fc1_out
        # print(fc1_out.size())
        output = torch.relu(self.fc2(fc1_out))
        # print(output.size())
        return output


class NET(nn.Module):  ### CTF 中的
    def __init__(self, input_size=16 * 3, hids_size=[]):
        super(NET, self).__init__()
        self.input_size = input_size
        self.hids_size = hids_size
        self.Modelist = nn.ModuleList()

        input_layer = nn.Linear(input_size, hids_size[0])
        output_layer = nn.Linear(hids_size[-1], 1)
        self.Modelist.append(input_layer)
        self.Modelist.append(nn.ReLU())
        for i in range(len(hids_size) - 1):
            self.Modelist.append(nn.Linear(hids_size[i], hids_size[i + 1]))
            self.Modelist.append(nn.ReLU())
        self.Modelist.append(output_layer)

    def forward(self, X):
        for model in self.Modelist:
            X = model(X)
        return X

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

class ConvECAMLP(nn.Module):
    def __init__(self, shape, rank, nc, device):
        super(ConvECAMLP, self).__init__()
        self.device = device
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.rank = rank
        self.shape = shape
        # self.norm1 = nn.BatchNorm2d(1)
        # self.eca1 = eca_layer(1, 3)  # 3,nc
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=(1, len(shape)), padding=0)
        self.eca = eca_layer(nc, 3)  # 3,nc
        #self.norm2 = nn.BatchNorm2d(nc)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(rank, 1), padding=0)
        #self.eca3 = eca_layer(nc, 3)  # 3,nc
        #self.norm=nn.BatchNorm2d(nc)
        #self.eca=eca_layer(nc,3)#3,nc
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.last_size = nc

    def forward(self, x):
        embeddings = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        # print(len(embeddings))
        # print(embeddings[0].size(),embeddings[1].size(),embeddings[2].size())
        # print(embeddings[0],embeddings[1],embeddings[2])
        concatenated = torch.cat(embeddings, dim=1)
        # print(concatenated.size())
        # reshaped = concatenated.view((concatenated.size(0),self.rank,len(self.shape),1))
        out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        # print(reshaped.size())

        #out = self.norm1(out)
        #out = self.eca2(out)#3,nc
        #out = torch.relu(out)

        out = self.conv1(out)
        #out = self.norm2(out)
        #out = torch.relu(out)

        #out = self.eca2(out)
        #out = self.norm2(out)
        #out = torch.relu(out)

        out = self.conv2(out)
        #out = self.norm2(out)
        #out = torch.relu(out)

        out = self.eca(out)
        #out = self.norm2(out)
        out = torch.relu(out)

        out = self.flatten(out)
        # print(flattened.size())
        out = torch.relu(self.fc1(out))
        # print(fc1_out.size())
        output = torch.relu(self.fc2(out))
        # print(output.size())
        return output


class ConvKAN(nn.Module):
    def __init__(self, shape, rank, nc, device):
        super(ConvKAN, self).__init__()
        self.device = device
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.rank = rank
        self.shape = shape
        # self.norm1 = nn.BatchNorm2d(1)
        # self.eca1 = eca_layer(1, 3)  # 3,nc

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=(1, len(shape)), padding=0)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(self.rank, 1), padding=0)

        self.flatten = nn.Flatten()
        #self.fc1 = nn.Linear(nc, nc)
        #self.fc2 = nn.Linear(nc, 1)

        #self.fc1 = FastKAN([nc, 1])
        self.fc1 = FastKANLayer(nc, 1)

        self.last_size = nc

    def forward(self, x):
        embeddings = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        # print(len(embeddings))
        # print(embeddings[0].size(),embeddings[1].size(),embeddings[2].size())
        # print(embeddings[0],embeddings[1],embeddings[2])
        concatenated = torch.cat(embeddings, dim=1)
        # print(concatenated.size())
        # reshaped = concatenated.view((concatenated.size(0),self.rank,len(self.shape),1))
        out = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        # print(reshaped.size())

        #out = self.norm1(out)
        #out = self.eca2(out)#3,nc
        #out = torch.relu(out)
        #print(out.size())
        out = self.conv1(out)
        #out = self.norm2(out)
        #out = torch.relu(out)

        #out = self.eca2(out)
        #out = self.norm2(out)
        out = torch.relu(out)
        #print(out.size())
        out = self.conv2(out)
        #out = self.norm2(out)
        #out = torch.relu(out)

        #out = self.eca2(out)
        #out = self.norm2(out)
        out = torch.relu(out)
        #print(out.size())
        #out = self.pool(out)
        #print(out.size())
        out = self.flatten(out)
        #print(out.size())
        # print(flattened.size())
        out = torch.relu(self.fc1(out))
        # print(fc1_out.size())
        #out = torch.relu(self.fc2(out))
        # print(output.size())
        return out

class CostcoModel(nn.Module):
    def __init__(self, shape, rank, nc, device):
        super(CostcoModel, self).__init__()
        self.device = device
        self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.rank=rank
        self.shape=shape
        # self.concat = nn.Concatenate(dim=1)
        # self.reshape = nn.Reshape((rank, len(shape), 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=(1,len(shape)), padding=0)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(rank, 1), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.last_size = nc

    def forward(self,x):
        embeddings = [self.embeds[i](x[i]).view(-1,1,self.rank) for i in range(len(x))]#利用nn.Embedding生成初始化的M C D
        # embeds = [
        #     self.embeds(output_dim=rank, input_dim=shape[i])(inputs[i])
        #     for i in range(len(shape))
        # ]
        # print(len(embeddings))
        # print(embeddings[0].size(),embeddings[1].size(),embeddings[2].size())
        # print(embeddings[0],embeddings[1],embeddings[2])

        concatenated = torch.cat(embeddings,dim=1)
        #print(concatenated.size())
        #reshaped = concatenated.view((concatenated.size(0),self.rank,len(self.shape),1))
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

class newCostcoModel(nn.Module):
    def __init__(self, shape, rank, nc, device):
        super(newCostcoModel, self).__init__()
        self.device = device
        self.embeds = nn.ModuleList(
            [nn.Embedding(num_embeddings=shape[i], embedding_dim=rank) for i in range(len(shape))])
        self.rank = rank
        self.shape = shape
        # self.concat = nn.Concatenate(dim=1)
        # self.reshape = nn.Reshape((rank, len(shape), 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nc, kernel_size=(rank, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=(1, len(shape)), padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.last_size = nc

    def forward(self, x):
        embeddings = [self.embeds[i](x[i]).view(-1, 1, self.rank) for i in range(len(x))]  # 利用nn.Embedding生成初始化的M C D
        # embeds = [
        #     self.embeds(output_dim=rank, input_dim=shape[i])(inputs[i])
        #     for i in range(len(shape))
        # ]
        # print(len(embeddings))
        # print(embeddings[0].size(),embeddings[1].size(),embeddings[2].size())
        # print(embeddings[0],embeddings[1],embeddings[2])

        concatenated = torch.cat(embeddings, dim=1)
        # print(concatenated.size())
        # reshaped = concatenated.view((concatenated.size(0),self.rank,len(self.shape),1))
        reshaped = concatenated.view((concatenated.size(0), 1, self.rank, len(self.shape)))
        # print(reshaped.size())
        conv1_out = torch.relu(self.conv1(reshaped))
        # print(conv1_out.size())
        conv2_out = torch.relu(self.conv2(conv1_out))
        # print(conv2_out.size())
        flattened = self.flatten(conv2_out)
        # print(flattened.size())
        fc1_out = torch.relu(self.fc1(flattened))
        self.intermediate = fc1_out
        # print(fc1_out.size())
        output = torch.relu(self.fc2(fc1_out))
        # print(output.size())
        return output


class NET(nn.Module):  ### CTF 中的
    def __init__(self, input_size=16 * 3, hids_size=[]):
        super(NET, self).__init__()
        self.input_size = input_size
        self.hids_size = hids_size
        self.Modelist = nn.ModuleList()

        input_layer = nn.Linear(input_size, hids_size[0])
        output_layer = nn.Linear(hids_size[-1], 1)
        self.Modelist.append(input_layer)
        self.Modelist.append(nn.ReLU())
        for i in range(len(hids_size) - 1):
            self.Modelist.append(nn.Linear(hids_size[i], hids_size[i + 1]))
            self.Modelist.append(nn.ReLU())
        self.Modelist.append(output_layer)

    def forward(self, X):
        for model in self.Modelist:
            X = model(X)
        return X


class MLP(nn.Module):
    def __init__(self, order, dimensionality, device, layers):
        super().__init__()

        assert (layers[0] % order == 0), "layers[0] (=order*embedding_dim) must be divided by the tensor order"
        self.device = device

        embedding_dim = int(layers[0] / order)

        self.embeddings = nn.ModuleList()
        for i in range(order):
            self.embeddings.append(torch.nn.Embedding(dimensionality[i], embedding_dim))

        # list of weight matrices
        self.fc_layers = nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        # final prediction layer
        self.last_size = layers[-1]
        self.output_layer = nn.Linear(layers[-1], 1)
        # torch.sigmoid(out)
        # self.output_layer=nn.Sigmoid()

    def forward(self, x):
        embeddings = [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))]
        # concatenate embeddings to form input
        x = torch.cat(embeddings, 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = self.relu(x)
            x = self.dropout(x)
        self.intermediate = x
        out = self.output_layer(x)
        del embeddings
        return out

    def predict(self, tensor, batch_size=1024):
        k = int(tensor.shape[0] / batch_size)
        final_output = torch.zeros(tensor.shape[0])
        for i in range(k + 1):
            st_idx, ed_idx = i * batch_size, (i + 1) * batch_size
            if ed_idx > tensor.shape[0]:
                ed_idx = tensor.shape[0]
            if st_idx >= ed_idx:
                break
            idx = torch.LongTensor(list(range(st_idx, ed_idx)))
            x = tensor[idx, :].clone().to(self.device)
            final_output[idx] = self(x).flatten().cpu().detach().clone()
            del x, self.intermediate
            torch.cuda.empty_cache()

        return final_output