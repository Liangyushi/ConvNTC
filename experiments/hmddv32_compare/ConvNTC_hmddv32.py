import sys
import os
# 添加模块所在的文件夹到 sys.path
folder_path = "src/"
sys.path.append(folder_path)

from hmddv32_data import GetData
from MCTD import Model
from ConvNTC import ConvNTC
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
# from utils import draw

tl.set_backend('pytorch')

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

def he_init_1(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode = 'fan_in', nonlinearity='relu')
        #torch.nn.init.kaiming_normal_(m.weight, a=0.01, mode = 'fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_test(model, criterion, lr, epochs, train_loader, val_loader, idxs_test, labels_test,k,k_folds,device):
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    min_val, min_test, min_epoch, final_model = 9999, 9999, 0, 0
    # 训练模型
    # loss_train_list = []
    # loss_test_list = []
    for epoch in tqdm(range(epochs)):
    #for epoch in range(epochs):
        ##训练
        model.train()
        train_loss, valid_loss = 0, 0
        # loss_train_list_batch = []
        for inputs in train_loader:
            optimizer.zero_grad()
            inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
            outputs = model(inputs_gpu)
            #print(inputs[-1].unsqueeze(1))
            loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
        train_loss /= len(train_loader.dataset)
        # loss_train_list.append(train_loss)

        # 验证模型
        model.eval()
        val_loss = 0.0
        for inputs in val_loader:
            with torch.no_grad():
                inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                outputs = model(inputs_gpu)
                loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                val_loss += loss.item() * len(inputs)
        val_loss /= len(val_loader.dataset)
        # loss_test_list.append(val_loss)

        # if epoch % 5 == 0:
        #     print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if min_val <= val_loss and epoch - min_epoch >= 10:
            break

        if min_val > val_loss:
            min_val = val_loss
            min_epoch = epoch
            # torch.save(Neural_Model, './best_model.pt')
            testModel = model

    # draw(loss_train_list, loss_test_list, str(k+1) + '-loss.png')
    # print('Finished Training.\nK-fold, Epoch, min val_loss ({},{},{})'.format(k, min_epoch, min_val))
    # 测试模型
    testModel.eval()
    with torch.no_grad():
        inputs_gpu = idxs_test.T.to(device)
        outputs = testModel(inputs_gpu)
        loss = criterion(outputs, labels_test.unsqueeze(1).to(device))
        print(f"Fold {k + 1}/{k_folds}, Test Loss: {loss:.6f}")
    return outputs,testModel

class Experiments(object):

    def __init__(self, drug_drug_data, model_name='NCTF', msi=10, times=10,folds=5,a=0.5,negs=1,
                 lr=0.001, epoch=150, batch_size=2048, nc=57,
                 kernel_size=[(1, 3), (57, 1)], dims=[1],
                 **kwargs):
        super().__init__()
        self.drug_drug_data = drug_drug_data
        self.model = Model(model_name)
        self.msi = msi
        self.times = times
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.channel = nc
        self.shape = drug_drug_data.X.shape
        self.kernel_size = kernel_size
        self.dims = dims
        self.folds = folds
        self.a = a
        self.negs = negs
        self.parameters = kwargs

    def CV_triplet(self):
        k_folds = self.folds
        fix_seed(2024)
        metrics_tensor_all = np.zeros((1, 7))
        avgmetrics_tensor_10 = np.zeros((1, 7))
        j = 0
        kname = ['kernel1', 'kernel2','kernel3']
        dname = ['dims1', 'dims2', 'dima3']
        kernel_sizeList = [[(1, len(self.shape)), (r, 1)], [(r, 1), (1, len(self.shape))]]  # our
        dimsList = [[1], [self.channel, 1], [self.channel, self.channel, 1]]  # pre层
        s1=kname[kernel_sizeList.index(self.kernel_size)]
        s2=dname[dimsList.index(self.dims)]
        df = pd.DataFrame(columns=['j', 'methods', 'times', 'folds', 'kernel', 'dims', 'aupr', 'auc', 'f1_score', 'accuracy',
                     'recall', 'specificity',
                     'precision'])
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
                idxs_train1, idxs_val, labels_train1, labels_val = train_test_split(idxs, vals, test_size=0.1)
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
                kernel_size = self.kernel_size  # 0-our 1-costco原始设置
                dims = self.dims
                lr = self.lr
                epochs = self.epoch
                batch_size = self.batch_size
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

                msi = [13]
                mnameList = ['ConvNTC']
                mname = mnameList[msi.index(self.msi)]
                print(mname)

                ##### 获取NCTF学习所得的因子矩阵 M, C, D #####
                ### end to end 学习M C D
                # train_tensor = np.array(self.drug_drug_data.X, copy=True)
                # trainpos_index = tuple(index_matrix[:, np.where(poscv == k)[0]])
                # train_tensor[trainpos_index] = 0
                # train_X = torch.tensor(train_tensor, dtype=torch.float32)
                # S1 = torch.tensor(self.drug_drug_data.S1, dtype=torch.float32)
                # S2 = torch.tensor(self.drug_drug_data.S2, dtype=torch.float32)
                # _, M, C, D = self.model()(train_X, S1, S2,
                #                           r=self.parameters['r'],
                #                           mu=self.parameters['mu'], eta=self.parameters['eta'],
                #                           alpha=self.parameters['alpha'], beta=self.parameters['beta'],
                #                           lam=self.parameters['lam'],
                #                           tol=self.parameters['tol'], max_iter=self.parameters['max_iter']
                #                           )
                # print('NCTF')

                ### 直接导入提前学习好的因子矩阵 
                fname='NCTF_embeds/'+str(self.negs) +'n/factors_'+str(i)+'_times_'+str(k)+'_fold.pkl'
                print(fname)
                with open(fname, 'rb') as f:  # Python 3: open(..., 'rb')
                    M, C, D = pickle.load(f)
                M = M.to(device)
                C = C.to(device)
                D = D.to(device)
                
                print(M.shape, C.shape, D.shape)
                
                ### 构建深度非线性模型 ### 定义损失函数 和 优化器
                if self.msi == 13:
                    Neural_Model = ConvNTC(shape, rank, nc, M, C, D, device, kernel_size,
                                                   dims=dims, act_func=nn.SiLU, dropout=0.0, input_dropout=0.0,alpha = self.a).to(device)

                # print(Neural_Model)
                Neural_Model.apply(he_init_1)
                criterion = nn.BCEWithLogitsLoss()
                outputs,testModel = train_test(Neural_Model, criterion, lr, epochs, train_loader, val_loader, idxs_test, labels_test, k, k_folds, device)

                ## 存储每折每次的预测和真实值
                fname='pred_score_pkl/'+mname+'_'+str(i)+'_times_'+str(k)+'_foldscores.pkl'
                print(fname)
                with open(fname, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([testModel,idxs_test,labels_test.cpu().numpy(),outputs.T[0].cpu().numpy()], f)

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
                
                ### 计算评价指标
                metrics = self.get_metrics_1(labels_test.cpu().numpy(), outputs.T[0].cpu().numpy())
                # print(metrics)
                metrics_tensor = metrics_tensor + metrics
                metrics_tensor_all = metrics_tensor_all + metrics
                # print(metrics)
                aupr, auc_value, f1_score, accuracy, recall, specificity, precision = metrics
                df.loc[j] = [j, mname, i, k, kernel_size, dims, aupr, auc_value, f1_score, accuracy, recall, specificity, precision]
                j = j + 1

            result = np.around(metrics_tensor / k_folds, decimals=4)
            print('Times:\t', i + 1, ':\t', result)
            avgmetrics_tensor_10 = avgmetrics_tensor_10 + result

        #print(self.a,str(self.a))
        sname = s1 + '_' + s2 + '_' + str(self.a)
        # fname = os.path.join('newablation/new3', mname + '_' + sname + '_hmddv32_5times5CV_1neg_results_bceheinit.csv')
        # print(fname)
        # df.to_csv(fname, index=False)  # index=False 表示不写入行索引
        fname = os.path.join('compareTF', mname + '_' + sname + '_hmddv3.2_'+str(self.negs)+'neg_results_bceheinit_fixed_new.csv')
        df.to_csv(fname, index=False)  # index=False 表示不写入行索引
        # print(j)
        # print(df)
        # print(metrics_tensor_all)
        results_1 = np.around(metrics_tensor_all / j, decimals=4)
        print('final:\t', results_1)
        # results_2 = np.around(avgmetrics_tensor_10 / self.times, decimals=4)
        # print('final:\t', results_2)
        return results_1

    def get_metrics_1(self, real_score, predict_score):
        real_score = np.mat(real_score)
        predict_score = np.mat(predict_score)
        # print(real_score)
        # print(real_score.shape)
        # print(predict_score)
        # print(predict_score.shape)
        np.random.seed(2024)
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        # sorted_predict_score = np.array(sorted(list(set(predict_score))))
        # print(sorted_predict_score)
        # print(sorted_predict_score.shape)
        # print(np.array(real_score).flatten())
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

        # print(real_score.T)
        # print(real_score.T.shape)
        # print(np.mat(real_score).T)
        # print(np.mat(real_score).T.shape)
        # print(predict_score_matrix.shape)
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


if __name__ == '__main__':
    fix_seed(2024)
    ### 导入数据
    since = time.time()
    ###循环次数
    times = 5
    ## 设置参数
    lr = 0.0001  ## 设置均不同
    batch_size = 256 #1024
    epoch = 500
    shape = 3
    r = 57
    nc = int(2*r)
    kernel_sizeList = [[(1, shape), (r, 1)], [(r, 1), (1, shape)]]  # our
    dimsList = [[1], [nc, 1], [nc, nc, 1]]  # pre层
    msiList = [13]
    mnameList = ['ConvNTC']
    df = pd.DataFrame(columns=['neg','aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity', 'precision'])
    i = 0
    kernel_size = kernel_sizeList[0]
    folds = 5
    mu,eta,alpha,beta,lam=0.75,0.125,0.25,0.25,0.001
    a=0.5
    signal = 11
    for msi in [13]:
        mname = mnameList[msiList.index(msi)]
        print(mname)
        dims = dimsList[0]# one-layer
        #for neg in [1,2,4,6,8,10]:
        for neg in [1]:
            folder = 'data/hmddv32_neg/'+str(neg)+'n'
            drug_drug_data = GetData(miRNA_num=351, disease_num=325,filefolder=folder,signal=signal,neg=neg)
            since1 = time.time()
            experiment = Experiments(drug_drug_data, model_name='MCTD', msi=msi, times=times, folds=folds,a=a,negs=neg,
                                     lr=lr, epoch=epoch, batch_size=batch_size, nc=nc,
                                     kernel_size=kernel_size, dims=dims,
                                     r=r, mu=mu, eta=eta, alpha=alpha, beta=beta,lam=lam, tol = 1e-4, max_iter = 100)
    
            aupr, auc_value, f1_score, accuracy, recall, specificity, precision = experiment.CV_triplet()[0]
            df.loc[i] = [neg, aupr, auc_value, f1_score, accuracy, recall, specificity,precision]
            print(f"neg={neg}")
            print(f"auc={auc_value}\taupr={aupr}\tf1={f1_score}\tacc={accuracy}\trecall={recall}\tspe={specificity}\tpre={precision}\n")
            i = i + 1
            time_elapsed1 = time.time() - since1
            print(time_elapsed1 // 60, time_elapsed1 % 60)

    df.to_csv('ConvNTC_negResults.csv',index=False)
    time_elapsed = time.time() - since
    print(time_elapsed // 60, time_elapsed % 60)