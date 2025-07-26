import csv
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

class GetData(object):
    def __init__(self, miRNA_num=624, disease_num=950,filefolder='newmmd_10times_5cv',signal=11,neg=10):
        super().__init__()
        self.miRNA_num = miRNA_num
        self.disease_num = disease_num
        self.folder = filefolder
        self.signal =signal
        self.neg = neg
        # self.S1, self.S2, self.X, self.CYP, self.NCYP = self.__get__DS3__data__()
        # self.X = self.CYP
        #self.S1, self.S2, self.X, self.posidx, self.poslabel, self.negidx, self.neglabel = self.__get__mmd_SimCombine_data__()
        #self.S1, self.S2, self.X, self.posidx, self.poslabel, self.negidx, self.neglabel = self.__get__mmd_SimSplit_data__()
        self.S1, self.S2, self.X, self.posidx, self.poscv, self.negidx, self.negcv = self.__get_SimCombine_data__()
        #self.S1, self.S2, self.X, self.posidx, self.poscv, self.negidx, self.negcv = self.__get__newmmd_SimSplit_data__()

    def __get_SimCombine_data__(self):
        tname1 = ["msSim","mtSim","seqSim"]
        tname2 = ["dsSim","dtSim","meshSim"]

        tname3 = ['miRNAfunSim','miRNAseqSim']
        tname4 = ['disMeshSim']

        if self.signal == 11: ## small hmddv3.2 第1，3维有约束，第2维和第1维对策的张量方法【第二个条件不一定】
            s1_type_name = tname1
            s2_type_name = tname2
        elif self.signal == 12: ## large hmddv4.0 第1，3维有约束，第2维和第1维对策的张量方法
            s1_type_name = tname3
            s2_type_name = tname4
        elif self.signal == 21: ## small hmddv3.2 第1，2维有约束的张量方法
            s1_type_name = tname1
            s2_type_name = tname1
        elif self.signal == 22: ## large hmddv4.0 第1，2维有约束的张量方法
            s1_type_name = tname3
            s2_type_name = tname3
           

        s1_all = []
        s2_all = []
        for name in s1_type_name:
            fname = os.path.join(self.folder, '{}.csv'.format(name))
            #df = pd.read_csv(self.folder + name + ".csv", sep='\t', header=None)
            df = pd.read_csv(fname, header=None)
            mat = df.values
            s1_all.append(mat)

        for name in s2_type_name:
            #df = pd.read_csv(self.folder + name + ".csv", sep='\t', header=None)
            fname = os.path.join(self.folder, '{}.csv'.format(name))
            df = pd.read_csv(fname, header=None)
            mat = df.values
            s2_all.append(mat)

        s1 = np.mean(np.stack(s1_all), axis=0)
        s2 = np.mean(np.stack(s2_all), axis=0)
        print(s1.shape)
        print(s2.shape)

        all_posIndex = []
        all_poscv = []
        all_negIndex = []
        all_negcv= []
        for i in range(0,10):
            ###导入10次5cv的数据
            ##pos
            fname = os.path.join(self.folder, 'pos_mmd_{}neg_{}.txt'.format(self.neg,i+1))
            mmd = np.loadtxt(fname)
            posIndex, poscv = torch.tensor(mmd[:, 1:], dtype=torch.int),torch.tensor(mmd[:, 0], dtype=torch.int)
            all_posIndex.append(posIndex)
            all_poscv.append(poscv)
            ###neg
            fname = os.path.join(self.folder, 'neg_mmd_{}neg_{}.txt'.format(self.neg,i+1))
            mmd = np.loadtxt(fname)
            negIndex, negcv = torch.tensor(mmd[:, 1:], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int)
            all_negIndex.append(negIndex)
            all_negcv.append(negcv)
            print(i)

        tensor_zeros = torch.zeros(self.miRNA_num, self.miRNA_num, self.disease_num)
        fname = os.path.join(self.folder, 'pos_mmd_{}neg_{}.txt'.format(self.neg,1))
        mmd = np.loadtxt(fname)
        posIndex, poscv = torch.tensor(mmd[:, 1:], dtype=torch.int),torch.tensor(mmd[:, 0], dtype=torch.int)
        for i in range(0, posIndex.size(0)):
            tensor_zeros[posIndex[i][0], posIndex[i][1], posIndex[i][2]] = 1

        x = tensor_zeros.numpy()
        # x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        # x = tensor_zeros
        #print(x)
        print(x.shape)
        print(sum(sum(sum(x))))
        return s1, s2, x, all_posIndex,all_poscv,all_negIndex,all_negcv


# if __name__ == '__main__':

# ####new mmd
   
#     #drug_drug_data = GetData(miRNA_num=351, disease_num=325,filefolder='newmmd_10times_5cv',signal=11)
#     #drug_drug_data = GetData(miRNA_num=351, disease_num=325,filefolder='newmmd_10times_5cv',signal=21)

# #### mmd 
#     #drug_drug_data = GetData(miRNA_num=624, disease_num=950,filefolder='mmd_10times_5cv_1negrandom',signal=12)
#     drug_drug_data = GetData(miRNA_num=624, disease_num=950,filefolder='mmd_10times_5cv_1negrandom',signal=22)