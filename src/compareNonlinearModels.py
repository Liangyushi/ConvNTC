import torch
from torch import optim, nn

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

class DeepSynergy_new(nn.Module):
    def __init__(self, shape, rank, input_size, X_tr,X_val,X_train,X_test, act_func=nn.ReLU(), dropout=0.5, input_dropout=0.2,
                 dims=[8182, 4096, 1]):
        super(DeepSynergy_new, self).__init__()
        self.layers = nn.ModuleList()  # 用于存储网络层
        self.flatten = nn.Flatten()
        self.rank = rank
        self.input_size = input_size
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
        return X

class DTF_new(nn.Module):
    def __init__(self, shape, rank, input_size, embeds=[], nn_struc=[2048, 1024, 512], input_dp=0.2, first_dp=0.5,
                 second_dp=0.5):
        super(DTF_new, self).__init__()
        self.embeds = embeds
        self.layers = nn.ModuleList()  # 用于存储网络层
        self.flatten = nn.Flatten()
        self.rank = rank
        self.input_size = input_size

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
        for layer in self.layers:
            x = layer(x)
        return x
