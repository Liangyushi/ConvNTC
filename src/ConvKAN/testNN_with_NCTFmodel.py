from DataCombine_ALL import GetData
from ourMethod_gpu import Model
from NCTF_NN import NCTF_CostcoModel, CostcoModel, NET, NCTF_newCostcoModel, newCostcoModel,ConvECA,KANConv_MLP
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
#from utils import draw
#from KANConvs_MLP import KANC_MLP

tl.set_backend('pytorch')


class Experiments(object):

    def __init__(self, drug_drug_data, model_name='NCTF', msi=10, times=10, lr=0.001, epoch=150, batch_size=2048, nc=67,
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
        self.parameters = kwargs

    def CV_triplet(self):
        k_folds = 5
        np.random.seed(2024)
        metrics_tensor_all = np.zeros((1, 7))
        avgmetrics_tensor_10 = np.zeros((1, 7))
        j = 0
        df = pd.DataFrame(
            columns=['j', 'times', 'folds', 'aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity',
                     'precision'])
        for i in range(self.times):
            index_matrix = self.drug_drug_data.posidx[i].numpy().T
            poscv = self.drug_drug_data.poscv[i].numpy()
            neg_matrix = self.drug_drug_data.negidx[i].numpy().T
            negcv = self.drug_drug_data.negcv[i].numpy()
            metrics_tensor = np.zeros((1, 7))

            for k in range(k_folds):
                # train_tensor = np.array(self.drug_drug_data.X, copy=True)
                # trainpos_index = tuple(index_matrix[:, np.where(poscv == k)[0]])
                # train_tensor[trainpos_index] = 0
                #
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
                # print(M.shape, C.shape, D.shape)

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
                idxs_train1, idxs_val, labels_train1, labels_val = train_test_split(idxs, vals, test_size=0.1,
                                                                                    random_state=2024)
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

                ### 构建深度非线性模型
                if self.msi==10:
                    Neural_Model = CostcoModel(shape, rank, nc, device).to(device)
                elif self.msi==20:
                    Neural_Model = newCostcoModel(shape, rank, nc, device).to(device) #Costco
                elif self.msi==11:
                    train_tensor = np.array(self.drug_drug_data.X, copy=True)
                    trainpos_index = tuple(index_matrix[:, np.where(poscv == k)[0]])
                    train_tensor[trainpos_index] = 0
                    train_X = torch.tensor(train_tensor, dtype=torch.float32)
                    S1 = torch.tensor(self.drug_drug_data.S1, dtype=torch.float32)
                    S2 = torch.tensor(self.drug_drug_data.S2, dtype=torch.float32)
                    _, M, C, D = self.model()(train_X, S1, S2,
                                              r=self.parameters['r'],
                                              mu=self.parameters['mu'], eta=self.parameters['eta'],
                                              alpha=self.parameters['alpha'], beta=self.parameters['beta'],
                                              lam=self.parameters['lam'],
                                              tol=self.parameters['tol'], max_iter=self.parameters['max_iter']
                                              )
                    print('NCTF')
                    print(M.shape, C.shape, D.shape)
                    Neural_Model = NCTF_CostcoModel(shape, rank, nc, M, C, D, device).to(device)
                elif self.msi == 30:
                    Neural_Model = ConvECA(shape, rank, nc, device).to(device)
                elif self.msi == 40:
                    Neural_Model = KANConv_MLP(shape, rank, nc, device).to(device)

                #Neural_Model = NCTF_CostcoModel(shape, rank, nc, M, C, D, device).to(device)
                #Neural_Model = CostcoModel(shape, rank, nc, device).to(device)
                #Neural_Model = newCostcoModel(shape, rank, nc, device).to(device)
                #Neural_Model = ConvECA(shape, rank, nc, device).to(device)

                ### 定义损失函数 和 优化器
                criterion = nn.MSELoss()
                optimizer = optim.Adam(Neural_Model.parameters(), lr=lr)

                min_val, min_test, min_epoch, final_model = 9999, 9999, 0, 0
                # valbest_auc, valbest_aupr = 0, 0
                # 训练模型
                loss_train_list = []
                loss_test_list = []
                for epoch in range(epochs):
                    ##训练
                    Neural_Model.train()
                    train_loss, valid_loss = 0, 0
                    # loss_train_list_batch = []
                    for inputs in train_loader:
                        optimizer.zero_grad()
                        inputs_gpu = [tensor.to(device) for tensor in inputs[:-1]]
                        outputs = Neural_Model(inputs_gpu)
                        # print(outputs)
                        # print(inputs[-1])
                        # print(inputs[-1].unsqueeze(1))
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
                            outputs = Neural_Model(inputs_gpu)
                            loss = criterion(outputs, inputs[-1].unsqueeze(1).to(device))
                            val_loss += loss.item() * len(inputs)
                    val_loss /= len(val_loader.dataset)
                    loss_test_list.append(val_loss)

                    if epoch % 5 == 0:
                        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                    if min_val <= val_loss and epoch - min_epoch >= 10:
                        break

                    if min_val > val_loss:
                        min_val = val_loss
                        min_epoch = epoch
                        # torch.save(Neural_Model, './best_model.pt')
                        testModel = Neural_Model

                # draw(loss_train_list, loss_test_list, str(k+1) + '-loss.png')
                print('Fold-k Finished Training.\nK-fold, Epoch, min val_loss ({},{},{})'.format(k, min_epoch, min_val))

                # testModel=torch.load('./best_model.pt')
                # 测试模型，获得测试集中的预测值
                # Neural_Model.eval()
                testModel.eval()
                with torch.no_grad():
                    # print(idxs_test.T)
                    # inputs_gpu = [tensor.to(device) for tensor in idxs_test[:-1]]
                    inputs_gpu = idxs_test.T.to(device)
                    # print(labels_test.unsqueeze(1))
                    # outputs = Neural_Model(inputs_gpu)
                    outputs = testModel(inputs_gpu)
                    # print(outputs)
                    loss = criterion(outputs, labels_test.unsqueeze(1).to(device))
                    print(f"Fold {k + 1}/{k_folds}, Test Loss: {loss:.6f}\n")
                # print(labels_test)
                # print(outputs.T[0])

                metrics = self.get_metrics_1(labels_test.cpu().numpy(), outputs.T[0].cpu().numpy())
                print(metrics)
                metrics_tensor = metrics_tensor + metrics
                metrics_tensor_all = metrics_tensor_all + metrics
                # print(metrics)
                aupr, auc_value, f1_score, accuracy, recall, specificity, precision = metrics
                df.loc[j] = [j, i, k, aupr, auc_value, f1_score, accuracy, recall, specificity, precision]
                j = j + 1

            result = np.around(metrics_tensor / k_folds, decimals=4)
            print('Times:\t', i + 1, ':\t', result)
            avgmetrics_tensor_10 = avgmetrics_tensor_10 + result

        # fname = os.path.join('hmddv3.2', 'TDRC_hmddv3.2_withR30_10times5CV_results.csv')
        # df.to_csv(fname, index=False)  # index=False 表示不写入行索引
        print(j)
        # print(df)
        # print(metrics_tensor_all)
        results_1 = np.around(metrics_tensor_all / j, decimals=4)
        print('final:\t', results_1)
        results_2 = np.around(avgmetrics_tensor_10 / self.times, decimals=4)
        print('final:\t', results_2)
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

    def get_metrics_2(self, real_score, predict_score):
        np.random.seed(2024)
        # trues, preds =  np.array(real_score).flatten(),np.array(predict_score).flatten()
        print(predict_score)
        trues, preds = real_score, expit(predict_score)
        print(preds)
        # print(preds, trues)
        fpr1, tpr1, thresholds1 = roc_curve(trues, preds, pos_label=1)
        auc_value = auc(fpr1, tpr1)
        # print(thresholds1)
        # auc_value = roc_auc_score(trues, preds)
        precision, recall, thresholds2 = precision_recall_curve(trues, preds, pos_label=1)
        precision = precision + np.finfo(float).tiny  # 添加极小值防止出现0
        recall = recall + np.finfo(float).tiny  # 添加极小值防止出现0
        # print(thresholds2)

        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_f1_index = f1_scores.argmax()
        best_f1 = f1_scores[best_f1_index]
        best_threshold = thresholds2[best_f1_index]
        print('Best F1 Score:', best_f1)
        print('Best Threshold:', best_threshold)
        best_recall = recall[best_f1_index]
        best_precision = precision[best_f1_index]
        print(best_recall, best_precision)
        f1_scores = 2 * (best_precision * best_recall) / (best_precision + best_recall)
        print(f1_scores)
        print(best_recall, best_precision, f1_scores)

        # best_threshold = np.median(thresholds1) # 中位数

        aupr = average_precision_score(trues, preds, pos_label=1)
        preds1 = preds
        preds1[preds > best_threshold] = 1
        preds1[preds <= best_threshold] = 0

        labels = [1]
        TP, FP, FN, TN = 0, 0, 0, 0
        for label in labels:
            preds_tmp = np.array([1 if pred == label else 0 for pred in preds1])
            trues_tmp = np.array([1 if true == label else 0 for true in trues])
            # print(preds_tmp, trues_tmp)
            # print()
            # TP预测为1真实为1
            # TN预测为0真实为0
            # FN预测为0真实为1
            # FP预测为1真实为0
            TP += ((preds_tmp == 1) & (trues_tmp == 1)).sum()
            TN += ((preds_tmp == 0) & (trues_tmp == 0)).sum()
            FN += ((preds_tmp == 0) & (trues_tmp == 1)).sum()
            FP += ((preds_tmp == 1) & (trues_tmp == 0)).sum()
        print(TP, FP, FN, TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (TN + FP)
        npre = TN / (TN + FN)
        fpr = FP / (TN + FP)
        fnr = FN / (TP + FN)
        print(recall, precision, f1_score)

        return aupr, auc_value, f1_score, accuracy, recall, specificity, precision

    def get_metrics_3(self, real_score, predict_score):
        # fpr1, tpr1, thresholds = roc_curve(trues, preds, pos_label=1)
        # auc_value = auc(fpr1, tpr1)
        print(predict_score)
        trues, preds = real_score, expit(predict_score)
        print(preds)
        auc_value = roc_auc_score(trues, preds)
        # precision1, recall1, _ = precision_recall_curve(trues, preds, pos_label=1)
        aupr = average_precision_score(trues, preds, pos_label=1)
        preds1 = preds
        # preds1[preds1 > 0.5] = 1
        # preds1[preds1 <= 0.5] = 0
        preds1[preds1 > np.median(preds)] = 1
        preds1[preds1 <= np.median(preds)] = 0

        labels = [1]
        TP, FP, FN, TN = 0, 0, 0, 0
        for label in labels:
            preds_tmp = np.array([1 if pred == label else 0 for pred in preds1])
            trues_tmp = np.array([1 if true == label else 0 for true in trues])
            print(preds_tmp, trues_tmp)
            # print()
            # TP预测为1真实为1
            # TN预测为0真实为0
            # FN预测为0真实为1
            # FP预测为1真实为0
            TP += ((preds_tmp == 1) & (trues_tmp == 1)).sum()
            TN += ((preds_tmp == 0) & (trues_tmp == 0)).sum()
            FN += ((preds_tmp == 0) & (trues_tmp == 1)).sum()
            FP += ((preds_tmp == 1) & (trues_tmp == 0)).sum()
        print(TP, FP, FN, TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (TN + FP)
        npre = TN / (TN + FN)
        fpr = FP / (TN + FP)
        fnr = FN / (TP + FN)

        return aupr, auc_value, f1, accuracy, recall, specificity, precision


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
    drug_drug_data = GetData(miRNA_num=351, disease_num=325, filefolder='./newmmd_10times_5cv', signal=11)
    since = time.time()
    #print(drug_drug_data)
    #print(drug_drug_data.shape)
    ### Split
    experiment = Experiments(drug_drug_data, model_name='NCTF_torch_gpu_float32',
                             msi=40, times=1, lr=0.001, epoch=150, batch_size=2048, nc=30,
                             r=30, mu=0.75, alpha=0.25, eta=0.125, beta=0.25, lam=0.001, tol=1e-4, max_iter=100)

    print('\t'.join(map(str, experiment.CV_triplet()[0])))
    time_elapsed = time.time() - since
    print(time_elapsed // 60, time_elapsed % 60)

# if __name__ == '__main__':
#     fix_seed(2024)
#     drug_drug_data = GetData(miRNA_num=351, disease_num=325, filefolder='newmmd_10times_5cv', signal=11)
#     since = time.time()
#     # print(drug_drug_data)
#     # print(drug_drug_data.shape)
#     df = pd.DataFrame(
#         columns=['i', 'lr', 'epoch', 'batch_size', 'channel', 'aupr', 'auc', 'f1_score', 'accuracy', 'recall',
#                  'specificity', 'precision'])
#     i = 0
#     # lrList=np.array([0.1, 0.01, 0.001, 0.0001])
#     # lrList=[0.1, 0.01, 0.001, 0.0001]
#     r = 57
#     lrList = [0.01, 0.001, 0.0001]
#     epochList = [100, 150, 200, 300, 500]
#     batch_sizeList = [128, 256, 512, 1024, 2048]
#     ncList = [29, 57, 114]  # R=30 bestR=57
#     # ncList=[15,30,60]#R=30 bestR=57
#     for lr in lrList:
#         for epoch in epochList:
#             for batch_size in batch_sizeList:
#                 for nc in ncList:
#                     experiment = Experiments(drug_drug_data, model_name='NCTF_torch_gpu_float32', times=10, lr=lr,
#                                              epoch=epoch, batch_size=batch_size, nc=nc, r=r, mu=0.75, alpha=0.25,
#                                              eta=0.125, beta=0.25, lam=0.001, tol=1e-4, max_iter=100)
#                     aupr, auc, f1_score, accuracy, recall, specificity, precision = experiment.CV_triplet()[0]
#                     df.loc[i] = [i, lr, epoch, batch_size, nc, aupr, auc, f1_score, accuracy, recall, specificity,
#                                  precision]
#                     # experiment.CV_triplet()[0]
#                     print(f"i={i}\tlr={lr}\tepoch={epoch}\tbatch_size={batch_size}\tchannel={nc}")
#                     print(
#                         f"auc={auc}\taupr={aupr}\tf1={f1_score}\tacc={accuracy}\trecall={recall}\tspe={specificity}\tpre={precision}\n")
#                     print(i)
#                     i = i + 1
#
#     df.to_csv('parameterGrid_hmddv3.2_10times5CV_withR57_ourNN.csv', index=False)  # index=False 表示不写入行索引
#     print('\t'.join(map(str, experiment.CV_triplet()[0])))
#     time_elapsed = time.time() - since
#     print(time_elapsed // 60, time_elapsed % 60)
#     i = i + 1

