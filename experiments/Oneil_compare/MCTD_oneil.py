import sys
import os

# 添加模块所在的文件夹到 sys.path
folder_path = "/mnt/sda/liupei/NCTF_new/src/"
sys.path.append(folder_path)

# 导入模块
from drug_oneil_data import GetData
from MCTD import Model
# from compareNumpyMethod import Model
import numpy as np
import time
import random
from torch.backends import cudnn
import tensorly as tl
import torch
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from scipy.special import expit

tl.set_backend('pytorch')


class Experiments(object):

    def __init__(self, drug_drug_data, model_name='MCTD', times=5, folds=5, negs=1, **kwargs):
        super().__init__()
        self.drug_drug_data = drug_drug_data
        self.model = Model(model_name)
        self.parameters = kwargs
        self.times = times
        self.folds = folds
        self.negs = negs

    def CV_triplet(self):
        fix_seed(2024)
        k_folds = self.folds
        np.random.seed(2024)
        metrics_tensor_all = np.zeros((1, 7))
        # avgmetrics_tensor_10 = np.zeros((1, 7))
        j = 0
        df = pd.DataFrame(
            columns=['j', 'times', 'folds', 'aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity',
                     'precision'])
        # score = pd.DataFrame(columns=['j', 'times', 'folds','aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity', 'precision'])
        for i in range(self.times):
            index_matrix = self.drug_drug_data.posidx[i].numpy().T
            poscv = self.drug_drug_data.poscv[i].numpy()
            neg_matrix = self.drug_drug_data.negidx[i].numpy().T
            negcv = self.drug_drug_data.negcv[i].numpy()
            metrics_tensor = np.zeros((1, 7))
            since = time.time()
            for k in range(k_folds):
                train_tensor = np.array(self.drug_drug_data.X, copy=True)
                trainpos_index = tuple(index_matrix[:, np.where(poscv == k)[0]])
                train_tensor[trainpos_index] = 0
                # S1 = np.mat(self.drug_drug_data.S1)
                # S2 = np.mat(self.drug_drug_data.S2)

                # train_X = tl.tensor(train_tensor)
                # S1 = tl.tensor(self.drug_drug_data.S1)
                # S2 = tl.tensor(self.drug_drug_data.S2)

                train_X = torch.tensor(train_tensor, dtype=torch.float32)
                S1 = torch.tensor(self.drug_drug_data.S1, dtype=torch.float32)
                S2 = torch.tensor(self.drug_drug_data.S2, dtype=torch.float32)
                predict_tensor, M, C, D = self.model()(train_X, S1, S2,
                                                       r=self.parameters['r'],
                                                       mu=self.parameters['mu'], eta=self.parameters['eta'],
                                                       alpha=self.parameters['alpha'], beta=self.parameters['beta'],
                                                       lam=self.parameters['lam'],
                                                       tol=self.parameters['tol'], max_iter=self.parameters['max_iter']
                                                       # device=self.parameters['device']
                                                       # net=self.parameters['net']
                                                       # epoch=self.parameters['epoch'], batch_size=self.parameters['batch_size'],
                                                       # hids_size=self.parameters['hids_size'],
                                                       # lr=self.parameters['lr'], weight_decay=self.parameters['weight_decay'],
                                                       # k=k, label=np.array(self.drug_drug_data.X, copy=True), train_index=train_index,
                                                       # Y=np.array(self.drug_drug_data.X, copy=True)
                                                       )

                print(k, 'fold end!')
                fname = 'NCTF_embeds/factors_' + str(i) + '_times_' + str(k) + '_fold.pkl'
                print(fname)
                with open(fname, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([M, C, D], f)

                del M, C, D

                # testpos_index 和 posIndex_test是一样的
                posIndex_test = torch.tensor(index_matrix[:, np.where(poscv == k)[0]], dtype=torch.int).T
                negIndex_test = torch.tensor(neg_matrix[:, np.where(negcv == k)[0]], dtype=torch.int).T
                idxs_test = tuple(torch.cat((posIndex_test, negIndex_test), dim=0).numpy().T)
                # print(idxs_test)

                poslabel_test = torch.ones(posIndex_test.shape[0])
                neglabel_test = torch.zeros(negIndex_test.shape[0])
                labels_test = torch.cat((poslabel_test, neglabel_test), dim=0)

                ### 获得预测值
                preds = predict_tensor[idxs_test].flatten()
                # print(labels_test.cpu().numpy().shape, preds.shape)
                # metrics_tensor = metrics_tensor + self.get_metrics_1(labels_test.cpu().numpy(), preds)
                # metrics_tensor = metrics_tensor + self.get_metrics_2(labels_test.cpu().numpy(), preds)
                # metrics_tensor = metrics_tensor + self.get_metrics_3(labels_test.cpu().numpy(), preds)

                ## 存储每折每次的预测和真实值
                fname = 'pred_score_pkl/' + 'NCTF_ddi_' + str(i) + '_times_' + str(k) + '_foldscores.pkl'
                print(fname)
                with open(fname, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([predict_tensor, idxs_test, labels_test.cpu().numpy(), preds], f)

                results = pd.DataFrame({
                    'time': [i] * len(idxs_test[0]),  # 假设这是第 1 折
                    'fold': [k] * len(idxs_test[0]),  # 假设这是第 1 次
                    'm1': idxs_test[0],
                    'm2': idxs_test[1],
                    'd': idxs_test[2],
                    'true_label': labels_test.cpu().numpy(),
                    'pred_score': preds  # 假设 preds 是一个二维数组，取第二列作为预测概率
                })
                # 保存为 CSV 文件
                fname = 'pred_score_csv/' + 'NCTF_ddi_' + str(i) + '_times_' + str(k) + '_foldscores.csv'
                results.to_csv(fname, index=False)

                metrics = self.get_metrics_1(labels_test.cpu().numpy(), preds)
                metrics_tensor = metrics_tensor + metrics
                metrics_tensor_all = metrics_tensor_all + metrics
                # print(metrics[0])
                # print(metrics)
                aupr, auc_value, f1_score, accuracy, recall, specificity, precision = metrics
                df.loc[j] = [j, i, k, aupr, auc_value, f1_score, accuracy, recall, specificity, precision]
                j = j + 1

            result = np.around(metrics_tensor / k_folds, decimals=4)
            print('Times:\t', i + 1, ':\t', result)
            # avgmetrics_tensor_10 = avgmetrics_tensor_10 + result
            time_elapsed = time.time() - since
            print(time_elapsed // 60, time_elapsed % 60)

        fname = os.path.join('compareTF', 'NCTF_ddi_' + str(self.negs) + 'neg_results_new.csv')
        df.to_csv(fname, index=False)  # index=False 表示不写入行索引
        print(j)
        # print(df)
        # print(metrics_tensor_all)
        results_1 = np.around(metrics_tensor_all / j, decimals=4)
        print('final:\t', results_1)
        # results_2 = np.around(avgmetrics_tensor_10 / self.times, decimals=4)
        # print('final:\t',results_2)
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
    since = time.time()
    # print(drug_drug_data)
    # print(drug_drug_data.shape)
    ### Split
    times = 5
    folds = 5
    df = pd.DataFrame(columns=['neg', 'aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity', 'precision'])
    j = 0
    r = 122
    mu, eta, alpha, beta, lam = 0.5, 2, 0.125, 0.125, 0.001
    for neg in [1]:
        signal = 13
        miRNA_num = 38
        disease_num = 39
        folder = '/data/oneil_ddi5cv'
        drug_drug_data = GetData(miRNA_num=miRNA_num, disease_num=disease_num, filefolder=folder, signal=signal)
        experiment = Experiments(drug_drug_data, model_name='MCTD', times=times, folds=folds,
                                 negs=neg,
                                 r=r, mu=mu, alpha=alpha, eta=eta, beta=beta, lam=lam, tol=1e-4, max_iter=100)
        aupr, auc_value, f1_score, accuracy, recall, specificity, precision = experiment.CV_triplet()[0]
        df.loc[j] = [neg, aupr, auc_value, f1_score, accuracy, recall, specificity, precision]
        print(f"neg={neg}")
        print(
            f"auc={auc_value}\taupr={aupr}\tf1={f1_score}\tacc={accuracy}\trecall={recall}\tspe={specificity}\tpre={precision}\n")
        j = j + 1

    df.to_csv('MCTD_1negResults.csv', index=False)  # index=False 表示不写入行索引
    time_elapsed = time.time() - since
    print(time_elapsed // 60, time_elapsed % 60)
