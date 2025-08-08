load("D:/peiStudy/DDI-new/oneil_DeepSynergy/Deepsynergy_ddi_oneil_new.rda")
novelDrugpair <- read_excel("oneil_DeepSynergy/novelDrugpair.xlsx")
df<-as.data.frame(novelDrugpair)

d1<-unique(df$`Drug A`)
d2<-unique(df$`Drug B`)
d12<-union(d1,d2)#41
tmp1<-df[,c(1,2)]
tmp2<-df[,c(3,4)]
colnames(tmp1)<-c('drug','smile')
colnames(tmp2)<-c('drug','smile')
newdrug<-rbind(tmp1,tmp2)
newdrug<-newdrug[!duplicated(newdrug),]#41
newdrug<-newdrug[order(newdrug$drug),]
id=seq(38,78)
newdrug<-cbind(id,newdrug)

#load("D:/peiStudy/DDI/DTF_ddi.rda")
smiles <- read_csv("oneil_DeepSynergy/smiles.csv", col_names = FALSE)
smiles<-as.data.frame(smiles)
library(stringr)
smiles$X1 <- str_to_title(smiles$X1, locale = "en")
colnames(smiles)<-c('name','smiles')

drug_smiles<-merge(drug,smiles,by='name')
drug_smiles<-drug_smiles[order(drug_smiles$id),]
drug_smiles<-drug_smiles[,c(2,1,3)]

# s<-vector()
# for (i in 1:nrow(drug)) {
#   
#   s[i]=smiles$smiles[which(drug$name[i]==smiles$name)]
# }
# drug<-cbind(drug,s)
write.table(drug_smiles, file ="oneil_DeepSynergy/38drugsmile.csv", sep = ",", row.names = FALSE, 
            quote = FALSE, col.names = FALSE)

d1<-drug_smiles$name
d2<-newdrug$drug
d12<-intersect(d1,d2)#有四个重叠药物
# d12
# "Erlotinib" "Lapatinib" "Sorafenib" "Sunitinib"
d<-setdiff(d2,d1)#37

tmp1<-drug_smiles#38
tmp2<-newdrug[newdrug$drug %in% d,]#37

colnames(tmp1)<-c('id','drug','smile')
colnames(tmp2)<-c('id','drug','smile')
alldrug<-rbind(tmp1,tmp2)#75
id=seq(0,nrow(alldrug)-1)
alldrug$id<-id

tmp<-df
for (i in 1:nrow(alldrug)) {
  tmp[which(tmp[,1]==alldrug[i,2]),1]=alldrug[i,1]
  tmp[which(tmp[,3]==alldrug[i,2]),3]=alldrug[i,1]
}

for (i in 1:nrow(cell)) {
  tmp[which(tmp[,5]==cell[i,2]),5]=cell[i,1]
}
novelddi<-tmp[,c(1,3,5)]

write.table(alldrug, file ="oneil_DeepSynergy/75allDrugsmile.csv", sep = ",", row.names = FALSE, 
            quote = FALSE, col.names = FALSE)

write.table(novelddi, file ="oneil_DeepSynergy/novelddi.csv", sep = ",", row.names = FALSE, 
            quote = FALSE, col.names = FALSE)