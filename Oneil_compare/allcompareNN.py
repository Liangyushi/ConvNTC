import sys
import os
# 添加模块所在的文件夹到 sys.path
folder_path = "/mnt/sda/liupei/NCTF_new/src/"
sys.path.append(folder_path)

from DataCombine_drug_oneil import GetData
from ourMethod_gpu import Model
from newNCTF_NN_2 import NCTF_ConvKAN_16, NCTF_ConvKAN_16_noeca1, NCTF_ConvKAN_16_noeca2, NCTF_ConvKAN_16_noeca12
from newNCTF_NN_2 import NCTF_ConvMLP_16
from newNCTF_NN_2 import NCTF_ConvKAN_13, NCTF_ConvKAN_1, ConvKAN
from newNCTF_NN_2 import NCTF_ConvKAN_17, NCTF_ConvKAN_11, ConvKAN_1
from newNCTF_NN_2 import Costco
from newNCTF_NN_2 import CTF_DDI
from newNCTF_NN_2 import DeepSynergy_new, DTF_new
from compareNumpyMethod import Model as numpyModel
#from DataCombine_drug import GetData
import pandas as pd
import os
from torch.backends import cudnn
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import math
import time
import tensorly as tl
from tqdm import tqdm
import gzip


# from utils import draw

# tl.set_backend('pytorch')

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

def normalize_Deepsynergy(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        X[:,std2==0]=0
        return(X, means1, std1, means2, std2, feat_filt)      

def normalize_DTF(X, means1=None, std1=None, means2=None, std2=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    X = np.ascontiguousarray(X)
    if norm is None:
        return (X, means1, std1, feat_filt)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1
    if norm == 'norm':
        return(X, means1, std1)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        return(X, means1, std1, means2, std2)
        
class Experiments(object):

    def __init__(self, drug_drug_data, model_name='NCTF', msi=10, times=10, negs=1,
                 lr=0.001, epoch=150, batch_size=2048, nc=57,
                 **kwargs):
        super().__init__()
        self.drug_drug_data = drug_drug_data
        self.model = Model(model_name)
        self.numpyModel = numpyModel(model_name)
        self.msi = msi
        self.times = times
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.channel = nc
        self.negs = negs
        self.parameters = kwargs

    def CV_triplet(self):
        fix_seed(2024)
        k_folds = 5
        #np.random.seed(2024)
        metrics_tensor_all = np.zeros((1, 7))
        avgmetrics_tensor_10 = np.zeros((1, 7))
        j = 0
        df = pd.DataFrame(columns=['j', 'methods', 'times', 'folds', 'aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity','precision'])
        for i in range(self.times):
            index_matrix = self.drug_drug_data.posidx[i].numpy().T
            poscv = self.drug_drug_data.poscv[i].numpy()
            neg_matrix = self.drug_drug_data.negidx[i].numpy().T
            negcv = self.drug_drug_data.negcv[i].numpy()
            metrics_tensor = np.zeros((1, 7))

            for k in range(k_folds):
                ### Train data
                # posIndex_train = torch.tensor(torch.nonzero(train_X == 1), dtype=torch.int)
                posIndex_train = torch.tensor(index_matrix[:, np.where(poscv != k)[0]]).T
                negIndex_train = torch.tensor(neg_matrix[:, np.where(negcv != k)[0]]).T
                idxs_train = torch.cat((posIndex_train, negIndex_train), dim=0)
                # print(idxs_train)
                # print(idxs_train.shape)
                poslabel_train = torch.ones(posIndex_train.shape[0])
                neglabel_train = torch.zeros(negIndex_train.shape[0])
                labels_train = torch.cat((poslabel_train, neglabel_train), dim=0)
                # print(labels_train.shape)

                ### 划分验证集
                idxs = idxs_train.numpy().astype(int)
                vals = labels_train.numpy().astype(float)
                # print(idxs,vals)
                # print(idxs.shape, vals.shape)
                idxs_train1, idxs_val, labels_train1, labels_val = train_test_split(idxs, vals, test_size=0.1,random_state=2024)
                # idxs_train, idxs_val, labels_train, labels_val = train_test_split(idxs_train, labels_train, test_size=0.1)

                ### Test data
                posIndex_test = torch.tensor(index_matrix[:, np.where(poscv == k)[0]], dtype=torch.int).T
                negIndex_test = torch.tensor(neg_matrix[:, np.where(negcv == k)[0]], dtype=torch.int).T
                idxs_test = torch.cat((posIndex_test, negIndex_test), dim=0)
                # print(idxs_test)
                # print(idxs_test.shape)
                poslabel_test = torch.ones(posIndex_test.shape[0])
                neglabel_test = torch.zeros(negIndex_test.shape[0])
                labels_test = torch.cat((poslabel_test, neglabel_test), dim=0)
                # print(labels_test.shape)

                # 模型超参数
                shape = self.drug_drug_data.X.shape
                rank = self.parameters['r']
                nc = self.channel
                lr = self.lr
                epochs = self.epoch
                batch_size = self.batch_size
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                methodList = [20, 70, 80, 90]
                mnameList = ['Costco', 'DeepSynergy', 'DTF', 'CTF_DDI']

                ### 构建深度非线性模型 ### 定义损失函数 和 优化器
                if self.msi == 20:
                    mname = mnameList[methodList.index(self.msi)]
                    print(mname)

                    # ### 划分验证集
                    # idxs = idxs_train.numpy().astype(int)
                    # vals = labels_train.numpy().astype(float)
                    # # print(idxs,vals)
                    # # print(idxs.shape, vals.shape)
                    # idxs_train1, idxs_val, labels_train1, labels_val = train_test_split(idxs, vals, test_size=0.1,random_state=2024)
                    
                    # 创建数据加载器
                    idxs_train1 = torch.LongTensor(idxs_train1)
                    idxs_val = torch.LongTensor(idxs_val)
                    # idxs_test = torch.LongTensor(idxs_test)
                    labels_train1 = torch.FloatTensor(labels_train1)
                    labels_val = torch.FloatTensor(labels_val)
                    # labels_test = torch.FloatTensor(labels_test)
                    
                    train_dataset = TensorDataset(*[idxs_train1[:, i] for i in range(len(shape))], labels_train1)
                    val_dataset = TensorDataset(*[idxs_val[:, i] for i in range(len(shape))], labels_val)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    del train_dataset, val_dataset
                    
                    Neural_Model = Costco(shape, rank, nc, device).to(device)  # Costco
                    print(Neural_Model)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(Neural_Model.parameters(), lr=lr)

                    # 训练模型
                    min_val, min_test, min_epoch, final_model = 9999, 9999, 0, 0
                    loss_train_list = []
                    loss_test_list = []
                    for epoch in tqdm(range(epochs)):
                        ##训练
                        Neural_Model.train()
                        train_loss, valid_loss = 0, 0
                        # loss_train_list_batch = []
                        for inputs in train_loader:
                            #print(inputs)
                            optimizer.zero_grad()
                            inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                            #inputs_gpu = inputs[0].to(device)
                            outputs = Neural_Model(inputs_gpu)
                            loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() * len(inputs)
                        train_loss /= len(train_loader.dataset)
                        loss_train_list.append(train_loss)
    
                        # 验证模型
                        Neural_Model.eval()
                        val_loss = 0.0
                        for inputs in val_loader:
                            with torch.no_grad():
                                inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                                #inputs_gpu = inputs[0].to(device)
                                outputs = Neural_Model(inputs_gpu)
                                loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                                val_loss += loss.item() * len(inputs)
                        val_loss /= len(val_loader.dataset)
                        loss_test_list.append(val_loss)
    
                        # if epoch % 5 == 0:
                        #     print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
                        if min_val <= val_loss and epoch - min_epoch >= 10:
                            break
    
                        if min_val > val_loss:
                            min_val = val_loss
                            min_epoch = epoch
                            testModel = Neural_Model
                    
                    testModel.eval()
                    with torch.no_grad():
                        inputs_gpu = idxs_test.T.to(device)
                        #inputs_gpu = X_test.to(device)
                        outputs = testModel(inputs_gpu)
                        loss = criterion(outputs, labels_test.unsqueeze(1).to(device))
                        print(f"Fold {k + 1}/{k_folds}, Test Loss: {loss:.6f}")
                        print(outputs,labels_test)

                elif self.msi == 70:
                    mname = mnameList[methodList.index(self.msi)]
                    print(mname)
                    ### 获得三个维度的相似度特征
                    M = torch.tensor(self.drug_drug_data.S1, dtype=torch.float32)
                    C = torch.tensor(self.drug_drug_data.S1, dtype=torch.float32)
                    D = torch.tensor(self.drug_drug_data.S2, dtype=torch.float32)
                    # folder = '/mnt/sda/liupei/NCTF/newCode/data/hmddv32_neg/'+str(neg)+'n'
                    # M, C, D = getSimFeature(folder=folder, signal=11)
                    # M = torch.FloatTensor(M)
                    # C = torch.FloatTensor(C)
                    # D = torch.FloatTensor(D)
                    # inputSize= (M.shape[1]+C.shape[1]+D.shape[1])
                    # print(inputSize)
                    
                    print(idxs_train1.shape)
                    f_1 = M[idxs_train1.T[0]]
                    f_2 = C[idxs_train1.T[1]]
                    f_3 = D[idxs_train1.T[2]]
                    f123 = torch.cat([f_1,f_2,f_3], dim=1)
                    print(f123.shape)
                    X_tr=f123.numpy()

                    print(idxs_val.shape)
                    f_1 = M[idxs_val.T[0]]
                    f_2 = C[idxs_val.T[1]]
                    f_3 = D[idxs_val.T[2]]
                    f123 = torch.cat([f_1,f_2,f_3], dim=1)
                    print(f123.shape)
                    X_val=f123.numpy()

                    print(idxs_train.shape)
                    f_1 = M[idxs_train.T[0]]
                    f_2 = C[idxs_train.T[1]]
                    f_3 = D[idxs_train.T[2]]
                    f123 = torch.cat([f_1,f_2,f_3], dim=1)
                    print(f123.shape)
                    X_train=f123.numpy()

                    print(idxs_test.shape)
                    f_1 = M[idxs_test.T[0]]
                    f_2 = C[idxs_test.T[1]]
                    f_3 = D[idxs_test.T[2]]
                    f123 = torch.cat([f_1,f_2,f_3], dim=1)
                    print(f123.shape)
                    X_test=f123.numpy()
                    print(X_tr.shape,X_val.shape,X_train.shape,X_test.shape)
                    norm = 'tanh'    
                    ## training 
                    if norm == "tanh_norm":
                        X_tr, mean, std, mean2, std2, feat_filt = normalize_Deepsynergy(X_tr, norm=norm)
                        X_val, mean, std, mean2, std2, feat_filt = normalize_Deepsynergy(X_val, mean, std, mean2, std2, 
                                                                              feat_filt=feat_filt, norm=norm)
                    else:
                        X_tr, mean, std, feat_filt = normalize_Deepsynergy(X_tr, norm=norm)
                        X_val, mean, std, feat_filt = normalize_Deepsynergy(X_val, mean, std, feat_filt=feat_filt, norm=norm)

                    #print(X_tr.shape,X_val.shape)
                    ## testing    
                    if norm == "tanh_norm":
                        X_train, mean, std, mean2, std2, feat_filt = normalize_Deepsynergy(X_train, norm=norm)
                        X_test, mean, std, mean2, std2, feat_filt = normalize_Deepsynergy(X_test, mean, std, mean2, std2, 
                                                                              feat_filt=feat_filt, norm=norm)
                    else:
                        X_train, mean, std, feat_filt = normalize_Deepsynergy(X_train, norm=norm)
                        X_test, mean, std, feat_filt = normalize_Deepsynergy(X_test, mean, std, feat_filt=feat_filt, norm=norm)
                    
                    print(X_tr.shape,X_val.shape,X_train.shape,X_test.shape)

                    # 创建数据加载器
                    X_tr = torch.FloatTensor(X_tr)
                    X_val = torch.FloatTensor(X_val)
                    X_test = torch.FloatTensor(X_test)
                    labels_train1 = torch.FloatTensor(labels_train1)
                    labels_val = torch.FloatTensor(labels_val)
                    # labels_test_Score = torch.FloatTensor(labels_test_Score)
                    # labels_test = torch.FloatTensor(labels_test)
                    
                    train_dataset = TensorDataset(X_tr, labels_train1)
                    val_dataset = TensorDataset(X_val, labels_val)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    del train_dataset, val_dataset
                 
                    inputSize = X_tr.shape[1]
                    Neural_Model = DeepSynergy_new(shape, rank, inputSize, 
                                               X_tr,X_val,X_train,X_test,
                                               act_func=nn.ReLU(),dropout=0.5, input_dropout=0.2,
                                               dims=[8182, 4096, 1]).to(device)
                    # [8182, 4096, 1];[4096, 2048, 1]
                    print(Neural_Model)
                    criterion = nn.MSELoss()
                    Neural_Model.apply(he_init)
                    # optimizer = optim.Adam(Neural_Model.parameters(), lr=lr)
                    optimizer = optim.SGD(Neural_Model.parameters(), lr=lr, momentum=0.5)

                    # 训练模型
                    min_val, min_test, min_epoch, final_model = 9999, 9999, 0, 0
                    loss_train_list = []
                    loss_test_list = []
                    for epoch in tqdm(range(epochs)):
                        ##训练
                        Neural_Model.train()
                        train_loss, valid_loss = 0, 0
                        # loss_train_list_batch = []
                        for inputs in train_loader:
                            #print(inputs)
                            optimizer.zero_grad()
                            #inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                            inputs_gpu = inputs[0].to(device)
                            outputs = Neural_Model(inputs_gpu)
                            loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() * len(inputs)
                        train_loss /= len(train_loader.dataset)
                        loss_train_list.append(train_loss)
    
                        # 验证模型
                        Neural_Model.eval()
                        val_loss = 0.0
                        for inputs in val_loader:
                            with torch.no_grad():
                                #inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                                inputs_gpu = inputs[0].to(device)
                                outputs = Neural_Model(inputs_gpu)
                                loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                                val_loss += loss.item() * len(inputs)
                        val_loss /= len(val_loader.dataset)
                        loss_test_list.append(val_loss)
    
                        # if epoch % 5 == 0:
                        #     print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
                        if min_val <= val_loss and epoch - min_epoch >= 10:
                            break
    
                        if min_val > val_loss:
                            min_val = val_loss
                            min_epoch = epoch
                            testModel = Neural_Model
                    
                    testModel.eval()
                    #labels_test_Score = torch.FloatTensor(labels_test_Score)
                    labels_test = torch.FloatTensor(labels_test)
                    #print(labels_test)
                    with torch.no_grad():
                        #inputs_gpu = idxs_test.T.to(device)
                        inputs_gpu = X_test.to(device)
                        outputs = testModel(inputs_gpu)
                        #loss = criterion(outputs, labels_test_Score.unsqueeze(1).to(device))
                        loss = criterion(outputs, labels_test.unsqueeze(1).to(device))
                        print(f"Fold {k + 1}/{k_folds}, Test Loss: {loss:.6f}")
                        print(outputs,labels_test)
                        # label_pre=outputs
                        # label_pre[ label_pre < threshold ] = 0
                        # label_pre[ label_pre >= threshold ] = 1
                        # outputs[ outputs < 30 ] = 0
                        # outputs[ outputs >= 30 ] = 1
                        # labels_test [labels_test<30] = 0
                        # labels_test [labels_test >=30] =1
                        #print(outputs,labels_test)

                elif self.msi == 80:
                    mname = mnameList[methodList.index(self.msi)]
                    print(mname)
                    fname = os.path.join('/mnt/sda/liupei/NCTF_new/ddi_oneil/compare/bib_revised/1neg/cpwopt_1000R_drugoneil/output/'
                                         'embM_{}_times_{}_fold.csv'.format(i + 1, k + 1))
                    emb = pd.read_csv(fname, header=None)
                    M = emb.values
                    #M = torch.FloatTensor(M).to(device)
                    fname = os.path.join('/mnt/sda/liupei/NCTF_new/ddi_oneil/compare/bib_revised/1neg/cpwopt_1000R_drugoneil/output/'
                                         'embC_{}_times_{}_fold.csv'.format(i + 1, k + 1))
                    emb = pd.read_csv(fname, header=None)
                    C = emb.values
                    #C = torch.FloatTensor(C).to(device)
                    fname = os.path.join('/mnt/sda/liupei/NCTF_new/ddi_oneil/compare/bib_revised/1neg/cpwopt_1000R_drugoneil/output/'
                                         'embD_{}_times_{}_fold.csv'.format(i + 1, k + 1))
                    emb = pd.read_csv(fname, header=None)
                    D = emb.values
                    #D = torch.FloatTensor(D).to(device)
                    print(M.shape,C.shape,D.shape)

                    M = torch.FloatTensor(M)
                    C = torch.FloatTensor(C)
                    D = torch.FloatTensor(D)
                    ### 得到数据集的特征
                    print(idxs_train1.shape)
                    f_1 = M[idxs_train1.T[0]]
                    f_2 = C[idxs_train1.T[1]]
                    f_3 = D[idxs_train1.T[2]]
                    f123 = torch.cat([f_1,f_2,f_3], dim=1)
                    #print(f123)
                    X_tr=f123.numpy()

                    print(idxs_val.shape)
                    f_1 = M[idxs_val.T[0]]
                    f_2 = C[idxs_val.T[1]]
                    f_3 = D[idxs_val.T[2]]
                    f123 = torch.cat([f_1,f_2,f_3], dim=1)
                    #print(f123)
                    X_val=f123.numpy()

                    print(idxs_train.shape)
                    f_1 = M[idxs_train.T[0]]
                    f_2 = C[idxs_train.T[1]]
                    f_3 = D[idxs_train.T[2]]
                    f123 = torch.cat([f_1,f_2,f_3], dim=1)
                    #print(f123)
                    X_train=f123.numpy()

                    print(idxs_test.shape)
                    f_1 = M[idxs_test.T[0]]
                    f_2 = C[idxs_test.T[1]]
                    f_3 = D[idxs_test.T[2]]
                    f123 = torch.cat([f_1,f_2,f_3], dim=1)
                    #print(f123)
                    X_test=f123.numpy()
                    print(X_tr.shape,X_val.shape,X_train.shape,X_test.shape)

                    ### 数据特征标准化
                    norm = 'tanh'    
                    if norm == "tanh_norm":
                        X_tr, mean, std, mean2, std2 = normalize_DTF(X_tr, norm=norm)
                        X_val, mean, std, mean2, std2 = normalize_DTF(X_val, mean, std, mean2, std2,  norm=norm)
                        X_test, mean, std, mean2, std2 = normalize_DTF(X_test, mean, std, mean2, std2, norm=norm)    
                    else:
                        X_tr, mean, std = normalize_DTF(X_tr, norm=norm)
                        X_val, mean, std = normalize_DTF(X_val, mean, std, norm=norm)
                        X_test, mean, std = normalize_DTF(X_test, mean, std, norm=norm)
                    
                    print(X_tr.shape,X_val.shape,X_train.shape,X_test.shape)

                    # 创建数据加载器
                    X_tr = torch.FloatTensor(X_tr)
                    X_val = torch.FloatTensor(X_val)
                    X_test = torch.FloatTensor(X_test)
                    labels_train1 = torch.FloatTensor(labels_train1)
                    labels_val = torch.FloatTensor(labels_val)
                    print(labels_train1,labels_val)
                    # labels_test_Score = torch.FloatTensor(labels_test_Score)
                    # labels_test = torch.FloatTensor(labels_test)
                    
                    train_dataset = TensorDataset(X_tr, labels_train1)
                    val_dataset = TensorDataset(X_val, labels_val)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    del train_dataset, val_dataset
                    
                    ## 3000
                    #inputSize = rank * len(shape)
                    inputSize = X_tr.shape[1]
                    Neural_Model = DTF_new(shape, rank, inputSize, embeds=[M, C, D], nn_struc=[2048, 1024, 512],
                                       input_dp=0.2, first_dp=0.5, second_dp=0.5).to(device)
                    print(Neural_Model)
                    Neural_Model.apply(he_init)
                    criterion = nn.BCELoss()  # binary crossentropy loss
                    optimizer = optim.Adam(Neural_Model.parameters(), lr=lr, weight_decay=5e-4)
                    
                    # 训练模型
                    min_val, min_test, min_epoch, final_model = 9999, 9999, 0, 0
                    loss_train_list = []
                    loss_test_list = []
                    for epoch in tqdm(range(epochs)):
                        ##训练
                        Neural_Model.train()
                        train_loss, valid_loss = 0, 0
                        # loss_train_list_batch = []
                        for inputs in train_loader:
                            #print(inputs)
                            optimizer.zero_grad()
                            #inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                            inputs_gpu = inputs[0].to(device)
                            outputs = Neural_Model(inputs_gpu)
                            #print(inputs[-1])
                            loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() * len(inputs)
                        train_loss /= len(train_loader.dataset)
                        loss_train_list.append(train_loss)
    
                        # 验证模型
                        Neural_Model.eval()
                        val_loss = 0.0
                        for inputs in val_loader:
                            with torch.no_grad():
                                #inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                                inputs_gpu = inputs[0].to(device)
                                outputs = Neural_Model(inputs_gpu)
                                loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                                val_loss += loss.item() * len(inputs)
                        val_loss /= len(val_loader.dataset)
                        loss_test_list.append(val_loss)
    
                        # if epoch % 5 == 0:
                        #     print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
                        if min_val <= val_loss and epoch - min_epoch >= 10:
                            break
    
                        if min_val > val_loss:
                            min_val = val_loss
                            min_epoch = epoch
                            testModel = Neural_Model
                    
                    testModel.eval()
                    #labels_test_Score = torch.FloatTensor(labels_test_Score)
                    #labels_test = torch.FloatTensor(labels_test)
                    #print(labels_test)
                    with torch.no_grad():
                        #inputs_gpu = idxs_test.T.to(device)
                        inputs_gpu = X_test.to(device)
                        outputs = testModel(inputs_gpu)
                        #loss = criterion(outputs, labels_test_Score.unsqueeze(1).to(device))
                        loss = criterion(outputs, labels_test.unsqueeze(1).to(device))
                        print(f"Fold {k + 1}/{k_folds}, Test Loss: {loss:.6f}")
                        print(outputs,labels_test)
                
                elif self.msi == 90:
                    mname = mnameList[methodList.index(self.msi)]
                    print(mname)
                    # ### 划分验证集
                    # idxs = idxs_train.numpy().astype(int)
                    # vals = labels_train.numpy().astype(float)
                    # # print(idxs,vals)
                    # # print(idxs.shape, vals.shape)
                    # idxs_train1, idxs_val, labels_train1, labels_val = train_test_split(idxs, vals, test_size=0.1,random_state=2024)
                    
                    # 创建数据加载器
                    idxs_train1 = torch.LongTensor(idxs_train1)
                    idxs_val = torch.LongTensor(idxs_val)
                    # idxs_test = torch.LongTensor(idxs_test)
                    labels_train1 = torch.FloatTensor(labels_train1)
                    labels_val = torch.FloatTensor(labels_val)
                    # labels_test = torch.FloatTensor(labels_test)
                    
                    train_dataset = TensorDataset(*[idxs_train1[:, i] for i in range(len(shape))], labels_train1)
                    val_dataset = TensorDataset(*[idxs_val[:, i] for i in range(len(shape))], labels_val)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    del train_dataset, val_dataset

                    # tl.set_backend('numpy')
                    # train_tensor = np.array(self.drug_drug_data.X, copy=True)
                    # trainpos_index = tuple(index_matrix[:, np.where(poscv == k)[0]])
                    # train_tensor[trainpos_index] = 0
                    # S1 = np.mat(self.drug_drug_data.S1)
                    # S2 = np.mat(self.drug_drug_data.S2)
                    # _, M, C, D = self.numpyModel()(train_tensor, S1, S2,
                    #                                r=self.parameters['r'],
                    #                                mu=self.parameters['mu'], eta=self.parameters['eta'],
                    #                                alpha=self.parameters['alpha'], beta=self.parameters['beta'],
                    #                                lam=self.parameters['lam'],
                    #                                tol=self.parameters['tol'], max_iter=self.parameters['max_iter']
                    #                                )
                    # print('CTF')
                    # # print(M.shape, C.shape, D.shape)
                    # M = torch.FloatTensor(M).to(device)
                    # C = torch.FloatTensor(C).to(device)
                    # D = torch.FloatTensor(D).to(device)
                    # print(M.shape, C.shape, D.shape)
                    # print(M)

                    ### 直接导入提前学习好的因子矩阵 
                    fname='CTF_embeds/factors_'+str(i)+'_times_'+str(k)+'_fold.pkl'
                    #fname='/mnt/sda/liupei/NCTF/newCode/hmddv32/compare/neg/CTF_embeds/1n/factors_'+str(i)+'_times_'+str(k)+'_fold.pkl'
                    with open(fname, 'rb') as f:  # Python 3: open(..., 'rb')
                        M, C, D = pickle.load(f)
                    
                    M = torch.FloatTensor(M).to(device)
                    C = torch.FloatTensor(C).to(device)
                    D = torch.FloatTensor(D).to(device)
                    
                    Neural_Model = CTF_DDI(shape, rank, hids_size=[256, 256, 128], embeds=[M, C, D], device=device).to(device)
                    print(Neural_Model)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(Neural_Model.parameters(), lr=lr, weight_decay=5e-4)

                    # 训练模型
                    min_val, min_test, min_epoch, final_model = 9999, 9999, 0, 0
                    loss_train_list = []
                    loss_test_list = []
                    for epoch in tqdm(range(epochs)):
                        ##训练
                        Neural_Model.train()
                        train_loss, valid_loss = 0, 0
                        # loss_train_list_batch = []
                        for inputs in train_loader:
                            #print(inputs)
                            optimizer.zero_grad()
                            inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                            #inputs_gpu = inputs[0].to(device)
                            outputs = Neural_Model(inputs_gpu)
                            loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() * len(inputs)
                        train_loss /= len(train_loader.dataset)
                        loss_train_list.append(train_loss)
    
                        # 验证模型
                        Neural_Model.eval()
                        val_loss = 0.0
                        for inputs in val_loader:
                            with torch.no_grad():
                                inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                                #inputs_gpu = inputs[0].to(device)
                                outputs = Neural_Model(inputs_gpu)
                                loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                                val_loss += loss.item() * len(inputs)
                        val_loss /= len(val_loader.dataset)
                        loss_test_list.append(val_loss)
    
                        # if epoch % 5 == 0:
                        #     print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
                        if min_val <= val_loss and epoch - min_epoch >= 10:
                            break
    
                        if min_val > val_loss:
                            min_val = val_loss
                            min_epoch = epoch
                            testModel = Neural_Model
                    
                    testModel.eval()
                    with torch.no_grad():
                        inputs_gpu = idxs_test.T.to(device)
                        #inputs_gpu = X_test.to(device)
                        outputs = testModel(inputs_gpu)
                        loss = criterion(outputs, labels_test.unsqueeze(1).to(device))
                        print(f"Fold {k + 1}/{k_folds}, Test Loss: {loss:.6f}")
                        print(outputs,labels_test)

                ## 存储每折每次的预测和真实值
                # fname='pred_score_pkl/'+mname+'_'+str(i)+'_times_'+str(k)+'_foldscores.pkl'
                # print(fname)
                # with open(fname, 'wb') as f:  # Python 3: open(..., 'wb')
                #     pickle.dump([testModel,idxs_test,labels_test.cpu().numpy(),outputs.T[0].cpu().numpy()], f)

                # print(idxs_test.T)
                # print(idxs_test.T[0],len(idxs_test.T[0]))
                results = pd.DataFrame({
                    'time': [i] * len(idxs_test.T[0]),  # 假设这是第 1 折
                    'fold': [k] * len(idxs_test.T[0]),  # 假设这是第 1 次
                    'm1': idxs_test.T[0],
                    'm2': idxs_test.T[1],
                    'd': idxs_test.T[2],
                    'true_label': labels_test.cpu().numpy(),
                    'pred_score': outputs.T[0].cpu().numpy()  # 假设 preds 是一个二维数组，取第二列作为预测概率
                })
                # 保存为 CSV 文件
                fname='pred_score_csv/'+mname+'_'+str(i)+'_times_'+str(k)+'_foldscores.csv'
                results.to_csv(fname, index=False)
                
                metrics = self.get_metrics_1(labels_test.cpu().numpy(), outputs.T[0].cpu().numpy())
                metrics_tensor = metrics_tensor + metrics
                metrics_tensor_all = metrics_tensor_all + metrics
                aupr, auc_value, f1_score, accuracy, recall, specificity, precision = metrics
                df.loc[j] = [j, mname, i, k, aupr, auc_value, f1_score, accuracy, recall, specificity, precision]
                j = j + 1

            result = np.around(metrics_tensor / k_folds, decimals=4)
            print('Times:\t', i + 1, ':\t', result)
            avgmetrics_tensor_10 = avgmetrics_tensor_10 + result

        fname = os.path.join('compareTF', mname + '_ddi_'+str(self.negs)+'neg_results_0.5lam_fixed.csv')
        df.to_csv(fname, index=False)  # index=False 表示不写入行索引
        #print(j)
        results_1 = np.around(metrics_tensor_all / j, decimals=4)
        print('final:\t', results_1)
        # results_2 = np.around(avgmetrics_tensor_10 / self.times, decimals=4)
        # print('final:\t', results_2)
        return results_1

    def get_metrics_1(self, real_score, predict_score):
        real_score = np.mat(real_score)
        predict_score = np.mat(predict_score)
        np.random.seed(2024)
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        # print(ROC_dot_matrix)
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]

        return aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == '__main__':
    fix_seed(2024)
    ### 导入数据
    since = time.time()
    #df = pd.DataFrame(columns=['methods', 'times', 'aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity', 'precision'])
    methodList = [20, 70, 80, 90]
    mnameList = ['Costco', 'DeepSynergy','DTF', 'CTF_DDI']
    ## 设置参数
    lr = 0.00001  ## 设置均不同
    batch_size = 256
    epoch = 500
    shape = 3
    r = 122
    nc = int(2*r)
    times = 5
    folds = 5
    miRNA_num = 38
    disease_num = 39
    mu,eta,alpha,beta,lam=0.5,2,0.125,0.125,0.001
    #df = pd.DataFrame(columns=['neg','aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity', 'precision'])
    df_all = pd.DataFrame(columns=['mname','neg','aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity', 'precision'])
    j = 0
    for msi in [20,70,80,90]:
        mname = mnameList[methodList.index(msi)]
        print(mname, msi)
        since2 = time.time()
        df = pd.DataFrame(columns=['folds','aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity', 'precision'])
        i = 0
        #for neg in [1,2,4,6,8,10]:
        for neg in [1]:
            folder = '/mnt/sda/liupei/NCTF_new/data/oneil_ddi5cv/'
            print(neg)
            since1 = time.time()
            if msi == 20:
                ## Costco
                signal = 13
                drug_drug_data = GetData(miRNA_num=miRNA_num, disease_num=disease_num, filefolder=folder, signal=signal)
                experiment = Experiments(drug_drug_data, model_name='NCTF_torch_gpu_float32',msi=msi, times=times, folds=folds,negs=neg,
                                         lr=lr, epoch=epoch, batch_size=batch_size, nc=nc,
                                         r=r, mu=mu, eta=eta, alpha=alpha, beta=beta,lam=lam, tol=1e-4, max_iter=100)
            elif msi == 70:
                ### DeepSynergy DTF 论文设置的参数
                signal = 33
                drug_drug_data = GetData(miRNA_num=miRNA_num, disease_num=disease_num, filefolder=folder, signal=signal)
                experiment = Experiments(drug_drug_data, model_name='NCTF_torch_gpu_float32',
                                         msi=70, times=times, folds=folds,negs=neg,
                                         lr=0.00001, epoch=1000, batch_size=64, nc=nc,
                                         r=r, mu=mu, eta=eta, alpha=alpha, beta=beta,lam=lam, tol=1e-4, max_iter=100)
    
            elif msi == 80:
                ## DTF batch_size=128 r=1000
                ### DTF 论文设置的参数
                signal = 13
                drug_drug_data = GetData(miRNA_num=miRNA_num, disease_num=disease_num, filefolder=folder, signal=signal)
                experiment = Experiments(drug_drug_data, model_name='NCTF_torch_gpu_float32',
                                         msi=80, times=times, folds=folds,negs=neg,
                                         lr=0.00001, epoch=1000, batch_size=128, nc=nc,
                                         r=r, mu=mu, eta=eta, alpha=alpha, beta=beta,lam=lam, tol=1e-4, max_iter=100)
            elif msi == 90:
                ## CTF_DDI batch_size=1000 epoch=300 r=51 max_iter=200
                signal = 23  # 22
                drug_drug_data = GetData(miRNA_num=miRNA_num, disease_num=disease_num, filefolder=folder, signal=signal)
                # DTF 论文设置的参数
                experiment = Experiments(drug_drug_data, model_name='CTF',
                                         msi=90, times=times, folds=folds,negs=neg,
                                         lr=0.0001, epoch=300, batch_size=1000, nc=nc,
                                         r=r, mu=0.5, eta=0.2, alpha=0.5, beta=0.5,lam=0.5, tol=1e-6, max_iter=100)

            aupr, auc_value, f1_score, accuracy, recall, specificity, precision = experiment.CV_triplet()[0]
            df.loc[i] = [neg, aupr, auc_value, f1_score, accuracy, recall, specificity, precision]
            # experiment.CV_triplet()[0]
            print(f"i={i}\ttimes={times}\tmethods={mname}\tmsi={msi}\tneg={neg}")
            print(f"auc={auc_value}\taupr={aupr}\tf1={f1_score}\tacc={accuracy}\trecall={recall}\tspe={specificity}\tpre={precision}\n")
            i = i + 1
            time_elapsed1 = time.time() - since1
            print(time_elapsed1 // 60, time_elapsed1 % 60)
            
        fname = mname + '_Results_1neg.csv'
        df.to_csv(fname,index=False)  # index=False 表示不写入行索引
        time_elapsed2 = time.time() - since2
        print(time_elapsed2 // 60, time_elapsed2 % 60)
        
        df_all.loc[j] = [mname,neg, aupr, auc_value, f1_score, accuracy, recall, specificity, precision]
        j=j+1


    df_all.to_csv('allcompareNN_Results.csv',index=False)  # index=False 表示不写入行索引
    time_elapsed = time.time() - since
    print(time_elapsed // 60, time_elapsed % 60)