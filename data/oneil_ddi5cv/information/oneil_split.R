#load("D:/peiStudy/DDI-new/NCI/nci_new.rda")
load("D:/peiStudy/DDI-new/oneil_DeepSynergy/Deepsynergy_ddi_oneil_new.rda")
table(ddi$label)
pos<-ddi[which(ddi$label==1),]
neg<-ddi[which(ddi$label==0),]
#### 正负比例为 
# 66485/9008
# [1] 7.380662

# pos<-pos[,-1]
# pos<-pos[,-4]
# pos<-pos[!duplicated(pos),]#17978

##### 划分数据集
library(caret)
#n=nrow(posInd_mmd)
for(j in 1:10){
  
  set.seed(j)  # 设置随机种子以便结果可复现
  ###负样本
  #tmp <- neg
  # # 划分数据集
  tmp_neg <- neg
  folds <- sample(rep(0:4, length.out = nrow(neg)))
  #mmd_pos_neg<-rbind(posInd_mmd,tmp_neg) 
  tmp_neg$f<-folds
  tmp1<-tmp_neg[,c(7,1,2,3,4,5)]
  
  tmp_pos <- pos
  folds <- sample(rep(0:4, length.out = nrow(pos)))
  #mmd_pos_neg<-rbind(posInd_mmd,tmp_neg) 
  tmp_pos$f<-folds
  tmp2<-tmp_pos[,c(7,1,2,3,4,5)]
  
  fname=paste0("oneil_ddi5cv/neg_mmd_1neg_",j,'.txt')
  write.table(tmp1, file =fname, sep = "\t", row.names = FALSE, quote = FALSE, col.names = FALSE)
  
  fname=paste0("oneil_ddi5cv/pos_mmd_1neg_",j,'.txt')
  write.table(tmp2, file =fname, sep = "\t", row.names = FALSE, quote = FALSE, col.names = FALSE)
  
  gc()
}

# save(ddc,ddi,cell,drug,file='DTF_ddi.rda')
# 
# write.table(ddc, file ='ddi_original.csv', sep = "\t", row.names = FALSE, quote = FALSE, col.names = FALSE)