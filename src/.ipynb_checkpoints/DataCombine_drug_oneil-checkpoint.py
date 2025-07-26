import csv
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
import pickle 
import gzip

class GetData(object):
    def __init__(self, miRNA_num=624, disease_num=950,filefolder='ddi5cv',signal=13):
        super().__init__()
        self.miRNA_num = miRNA_num
        self.disease_num = disease_num
        self.folder = filefolder
        self.signal =signal
        #self.S1, self.S2, self.X, self.posidx, self.poscv, self.negidx, self.negcv = self.__get_SimCombine_data__()
        #self.S1, self.S2, self.X, self.synergyX, self.posidx, self.poscv, self.negidx, self.negcv, self.posScore, self.negScore = self.__get_drugWithSynergy_data__()
        if signal in [13,23,33]:
            self.S1, self.S2, self.X, self.synergyX, self.posidx, self.poscv, self.negidx, self.negcv, self.posScore, self.negScore, self.posTriInd, self.negTriInd = self.__get_drugWithSynergy_data__()
        else:
            self.feature, self.X, self.synergyX, self.posidx, self.poscv, self.negidx, self.negcv, self.posScore, self.negScore, self.posTriInd, self.negTriInd = self.__get_drug_data__()

    def __get_drug_data__(self):

        fname = os.path.join(self.folder, 'X.p.gz')
        file = gzip.open(fname, 'rb')
        f = pickle.load(file)
        file.close()
        print(f.shape)
        
        all_posIndex = []
        all_poscv = []
        all_negIndex = []
        all_negcv = []
        all_negSynergy = []
        all_posSynergy = []
        all_posTriInd = []
        all_negTriInd = []
        for i in range(0, 10):
            ###导入10次5cv的数据
            ##pos
            fname = os.path.join(self.folder, 'pos_mmd_1neg_{}.txt'.format(i + 1))
            mmd = np.loadtxt(fname)
            posIndex, poscv = torch.tensor(mmd[:, 2:5], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int)
            possynergyScore = torch.tensor(mmd[:, 5], dtype=torch.float32)
            posTriInd= torch.tensor(mmd[:, 1], dtype=torch.int)
            all_posIndex.append(posIndex)
            all_poscv.append(poscv)
            all_posSynergy.append(possynergyScore)
            all_posTriInd.append(posTriInd)
            ###neg
            fname = os.path.join(self.folder, 'neg_mmd_1neg_{}.txt'.format(i + 1))
            mmd = np.loadtxt(fname)
            negIndex, negcv = torch.tensor(mmd[:, 2:5], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int)
            negsynergyScore = torch.tensor(mmd[:, 5], dtype=torch.float32)
            negTriInd= torch.tensor(mmd[:, 1], dtype=torch.int)
            all_negIndex.append(negIndex)
            all_negcv.append(negcv)
            all_negSynergy.append(negsynergyScore)
            all_negTriInd.append(negTriInd)
            print(i)

        tensor_zeros = torch.zeros(self.miRNA_num, self.miRNA_num, self.disease_num,dtype=torch.float32)
        tensor_synergy = torch.zeros(self.miRNA_num, self.miRNA_num, self.disease_num,dtype=torch.float32)
        fname = os.path.join(self.folder, 'pos_mmd_1neg_{}.txt'.format(1))
        mmd = np.loadtxt(fname)
        posIndex, poscv, possynergyScore= (torch.tensor(mmd[:, 2:5], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int),
                                        torch.tensor(mmd[:, 5], dtype=torch.float32))
        #print(posIndex, poscv, possynergyScore)
        fname = os.path.join(self.folder, 'neg_mmd_1neg_{}.txt'.format(1))
        mmd = np.loadtxt(fname)
        negIndex, negcv, negsynergyScore= (torch.tensor(mmd[:, 2:5], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int),
                                        torch.tensor(mmd[:, 5], dtype=torch.float32))
        #print(negIndex, negcv, negsynergyScore)
        #print(possynergyScore[0].item())
        for i in range(0, posIndex.size(0)):
            tensor_zeros[posIndex[i][0], posIndex[i][1], posIndex[i][2]] = 1
            tensor_zeros[posIndex[i][1], posIndex[i][0], posIndex[i][2]] = 1
            tensor_synergy[posIndex[i][0], posIndex[i][1], posIndex[i][2]] = possynergyScore[i].item()
            tensor_synergy[posIndex[i][1], posIndex[i][0], posIndex[i][2]] = possynergyScore[i].item()

        for i in range(0, negIndex.size(0)):
            #tensor_zeros[posIndex[i][0], posIndex[i][1], posIndex[i][2]] = 1
            tensor_synergy[negIndex[i][0], negIndex[i][1], negIndex[i][2]] = negsynergyScore[i].item()
            tensor_synergy[negIndex[i][1], negIndex[i][0], negIndex[i][2]] = negsynergyScore[i].item()

        #print(tensor_synergy)
        x = tensor_zeros.numpy()
        print(x.shape)
        print(sum(sum(sum(x))))
        #print(x)

        s = tensor_synergy.numpy()
        print(s.shape)
        #print(s)
        return f, x, s, all_posIndex, all_poscv, all_negIndex, all_negcv,all_posSynergy,all_negSynergy,all_posTriInd,all_negTriInd

    def __get_drugWithSynergy_data__(self):

        # tname5 = ["Simdrug_cosine","Simdrug_jaccardAndCosine"]
        # tname6 = ["Simcell_cosine", "Simcell_spearman"]

        tname5 = ["Simdrug_cosine"]
        tname6 = ["Simcell_cosine"]

        tname7 = ["drug_feature"]
        tname8 = ["cell_feature"]
        if self.signal == 13:  # drug data 第1，3维有约束，第2维和第1维对策的张量方法
            s1_type_name = tname5
            s2_type_name = tname6
        elif self.signal == 23:  # drug data 第1，2维有约束的张量方法
            s1_type_name = tname5
            s2_type_name = tname5
        elif self.signal == 33:  # drug data 第1，2维有约束的张量方法
            s1_type_name = tname7
            s2_type_name = tname8

        s1_all = []
        s2_all = []
        for name in s1_type_name:
            fname = os.path.join(self.folder, '{}.csv'.format(name))
            # df = pd.read_csv(self.folder + name + ".csv", sep='\t', header=None)
            df = pd.read_csv(fname, header=None)
            mat = df.values
            s1_all.append(mat)

        for name in s2_type_name:
            # df = pd.read_csv(self.folder + name + ".csv", sep='\t', header=None)
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
        all_negcv = []
        all_negSynergy = []
        all_posSynergy = []
        all_posTriInd = []
        all_negTriInd = []
        for i in range(0, 10):
            ###导入10次5cv的数据
            ##pos
            fname = os.path.join(self.folder, 'pos_mmd_1neg_{}.txt'.format(i + 1))
            mmd = np.loadtxt(fname)
            posIndex, poscv = torch.tensor(mmd[:, 2:5], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int)
            possynergyScore = torch.tensor(mmd[:, 5], dtype=torch.float32)
            posTriInd= torch.tensor(mmd[:, 1], dtype=torch.int)
            all_posIndex.append(posIndex)
            all_poscv.append(poscv)
            all_posSynergy.append(possynergyScore)
            all_posTriInd.append(posTriInd)
            ###neg
            fname = os.path.join(self.folder, 'neg_mmd_1neg_{}.txt'.format(i + 1))
            mmd = np.loadtxt(fname)
            negIndex, negcv = torch.tensor(mmd[:, 2:5], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int)
            negsynergyScore = torch.tensor(mmd[:, 5], dtype=torch.float32)
            negTriInd= torch.tensor(mmd[:, 1], dtype=torch.int)
            all_negIndex.append(negIndex)
            all_negcv.append(negcv)
            all_negSynergy.append(negsynergyScore)
            all_negTriInd.append(negTriInd)
            print(i)

        tensor_zeros = torch.zeros(self.miRNA_num, self.miRNA_num, self.disease_num,dtype=torch.float32)
        tensor_synergy = torch.zeros(self.miRNA_num, self.miRNA_num, self.disease_num,dtype=torch.float32)
        fname = os.path.join(self.folder, 'pos_mmd_1neg_{}.txt'.format(1))
        mmd = np.loadtxt(fname)
        posIndex, poscv, possynergyScore= (torch.tensor(mmd[:, 2:5], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int),
                                        torch.tensor(mmd[:, 5], dtype=torch.float32))
        #print(posIndex, poscv, possynergyScore)
        fname = os.path.join(self.folder, 'neg_mmd_1neg_{}.txt'.format(1))
        mmd = np.loadtxt(fname)
        negIndex, negcv, negsynergyScore= (torch.tensor(mmd[:, 2:5], dtype=torch.int), torch.tensor(mmd[:, 0], dtype=torch.int),
                                        torch.tensor(mmd[:, 5], dtype=torch.float32))
        #print(negIndex, negcv, negsynergyScore)
        #print(possynergyScore[0].item())
        for i in range(0, posIndex.size(0)):
            tensor_zeros[posIndex[i][0], posIndex[i][1], posIndex[i][2]] = 1
            tensor_zeros[posIndex[i][1], posIndex[i][0], posIndex[i][2]] = 1
            tensor_synergy[posIndex[i][0], posIndex[i][1], posIndex[i][2]] = possynergyScore[i].item()
            tensor_synergy[posIndex[i][1], posIndex[i][0], posIndex[i][2]] = possynergyScore[i].item()

        for i in range(0, negIndex.size(0)):
            tensor_synergy[negIndex[i][0], negIndex[i][1], negIndex[i][2]] = negsynergyScore[i].item()
            tensor_synergy[negIndex[i][1], negIndex[i][0], negIndex[i][2]] = negsynergyScore[i].item()

        #print(tensor_synergy)
        x = tensor_zeros.numpy()
        print(x.shape)
        print(sum(sum(sum(x))))
        #print(x)

        s = tensor_synergy.numpy()
        print(s.shape)
        #print(s)
        return s1, s2, x, s, all_posIndex, all_poscv, all_negIndex, all_negcv,all_posSynergy,all_negSynergy,all_posTriInd,all_negTriInd


# if __name__ == '__main__':

# ####new mmd
   
#     #drug_drug_data = GetData(miRNA_num=351, disease_num=325,filefolder='newmmd_10times_5cv',signal=11)
#     #drug_drug_data = GetData(miRNA_num=351, disease_num=325,filefolder='newmmd_10times_5cv',signal=21)

# #### mmd
#     #drug_drug_data = GetData(miRNA_num=624, disease_num=950,filefolder='mmd_10times_5cv_1negrandom',signal=12)
#     #drug_drug_data = GetData(miRNA_num=624, disease_num=950,filefolder='mmd_10times_5cv_1negrandom',signal=22)

# #### ddi
#     drug_drug_data = GetData(miRNA_num=38, disease_num=39,filefolder='ddi5cv',signal=13)
# #     drug_drug_data = GetData(miRNA_num=38, disease_num=39,filefolder='ddi5cv',signal=23)
# # ####