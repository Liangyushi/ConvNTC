import torch
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset


class NET(nn.Module):
    def __init__(self, input_size=16 * 3, hids_size=[]):
        super(NET, self).__init__()
        self.input_size = input_size
        self.hids_size = hids_size
        self.Modelist = nn.ModuleList()

        input_layer = nn.Linear(input_size, hids_size[0])
        output_layer = nn.Linear(hids_size[-1], 1)
        self.Modelist.append(input_layer)
        self.Modelist.append(nn.ReLU())
        for i in range(len(hids_size)-1):
            self.Modelist.append(nn.Linear(hids_size[i], hids_size[i+1]))
            self.Modelist.append(nn.ReLU())
        self.Modelist.append(output_layer)

    def forward(self, X):
        for model in self.Modelist:
            X = model(X)
        return X


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

class IdDataset(Dataset):
    def __init__(self, lenth):
        self.ids = torch.tensor(range(lenth))

    def __getitem__(self, idx):
        return self.ids[idx]

    def __len__(self):
        return len(self.ids)


def draw(data_train, data_test, path):
    epochs = list(range(1, len(data_train) + 1))
    plt.figure()
    plt.plot(epochs, data_train, "b*-", label="train")
    plt.plot(epochs, data_test, "r*-", label="test")
    plt.legend()
    plt.savefig(path)
    plt.close()
