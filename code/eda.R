library(dplyr)
library(ggplot2)
library(tidyverse)
library(MASS)
library(highcharter)
library(klaR)
library(kknn)
library(caret)
library(e1071)
library(corrplot)

trainset <- read.table("~/MECD-1/2S/Data_Mining/Project1/DataMining/data/gisette_train.data", 
                    quote="\"", 
                    comment.char="", 
                    stringsAsFactors=FALSE)

train_labels <- read.table("~/MECD-1/2S/Data_Mining/Project1/DataMining/data/gisette_train.labels",
                            quote="\"", 
                            comment.char="", 
                            stringsAsFactors=FALSE) %>% 
  mutate(V1 = as.numeric(V1))%>% 
  rename("label" = V1)

testset <- read.table("~/MECD-1/2S/Data_Mining/Project1/DataMining/data/gisette_test.data", 
                    quote="\"", 
                    comment.char="", 
                    stringsAsFactors=FALSE)

test_labels <- read.table("~/MECD-1/2S/Data_Mining/Project1/DataMining/data/gisette_test.labels",
                           quote="\"", 
                           comment.char="", 
                           stringsAsFactors=FALSE)%>% 
  mutate(V1 = as.numeric(V1))%>% 
  rename("label" = V1)

train_full<-trainset %>% 
  cbind(train_labels)

train_means<-train_full %>% 
  group_by(label) %>% 
  summarise_all(mean)

train_var<-trainset %>% 
  summarise_all(var) %>% 
  t() %>% 
  as.data.frame() %>% 
  rownames_to_column("var_name") %>% 
  arrange(desc(V1))

hc_var<-highchart() %>% 
  hc_xAxis(categories = train_var$var_name) %>% 
  hc_chart(type = "bar") %>% 
  hc_add_series(train_var$V1, name = "Variance")

zero_var_features<-train_var %>% 
  filter(V1==0) %>% 
  .$var_name

trainset<-trainset %>% 
  dplyr::select(-one_of(zero_var_features))

testset<-testset %>% 
  dplyr::select(-one_of(zero_var_features))
# Baseline ----------------------------------------------------------------------------------------------

baseline_model<-train(label ~.,
                      data = train_full,
                      method = "glm",
                      trControl = trainControl(method = "cv", number = 5))
baseline_pred<-data.frame("prediction" = predict(baseline_model,
                       newdata = testset,
                       type = "raw")) %>% 
  mutate(prediction_class = ifelse(prediction>0,1,-1))

baseline_metrics<-confusionMatrix(as.factor(baseline_pred$prediction_class),
                                  as.factor(test_labels$label))

# PCA ---------------------------------------------------------------------------------------------------

pca<-prcomp(trainset)

pca_var<-pca$sdev^2
prop_var<-pca_var/sum(pca_var)

plot(prop_var, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")


plot(cumsum(prop_var), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

test_pca<-predict(pca,
                  newdata = testset)

### PCA 1400 components --------------------------------------------------------------------------------

var_1400<-max(cumsum(prop_var[1:1400]))

train_pca1400<-cbind(train_labels,pca$x) %>% 
  dplyr::select(1:1401)

test_pca1400<-test_pca %>%
  as.data.frame() %>% 
  dplyr::select(1:1400)

glm_pca_1400<-train(label ~.,
                    data = train_pca1400,
                    method = "glm",
                    trControl = trainControl(method = "cv", number = 5))

glm_pca_1400_pred<-data.frame("prediction" = predict(glm_pca_1400,
                                                 newdata = test_pca1400,
                                                 type = "raw")) %>% 
  mutate(prediction_class = ifelse(prediction>0,1,-1))

glm_pca_1400_metrics<-confusionMatrix(as.factor(glm_pca_1400_pred$prediction_class),
                                  as.factor(test_labels$label))

# LDA --------------------------------------------------------------------------------------------------

normalize<-trainset %>% 
  preProcess(method = c("center", "scale"))
train_normal<-predict(normalize,
                      newdata = trainset) %>% 
  cbind(train_labels)
test_normal<-predict(normalize,
                     newdata = testset)

lda<-lda(label ~.,
         data = train_normal)

lda_pred<-predict(lda,
                  newdata = test_normal)

plot_lda<-plot(lda)

lda_metrics<-confusionMatrix(lda_pred$class,
                             as.factor(test_labels$label))
### Examples Partition Plots
partimat(train_normal[,c(5,1000)], grouping=as.factor(train_normal$label), method="lda")

# QDA (reDo - error)--------------------------------------------------------------------------------------------------

# qda<-qda(label ~.,
#          data = train_normal)

# KNN ----------------------------------------------------------------------------------------------------------------

get_knn<-function(train,test,test_labels,k){
  
  knn<-kknn(label ~.,train,test,2,k = k)
  
  pred<-data.frame("pred" = fitted(knn)) %>% 
    mutate(pred_class = ifelse(pred>0,1,-1))
  
  accuracy<-sum(pred$pred_class == test_labels$label)/length(test_labels$label)
  
  return(list(pred,accuracy))
}

# knn_3<-kknn(label ~.,train_full,testset,2,k = 3)
# 
# knn3_pred<-data.frame("pred" = fitted(knn_3)) %>% 
#   mutate(pred_class = as.factor(ifelse(pred>0,1,-1)))
# knn3_metrics<-confusionMatrix(knn3_pred$pred_class,
#                               as.factor(test_labels$label))
k_values<-1:15
knn_res<-list()

for(k in k_values){
  knn_res[[k]]<-get_knn(train_full,testset,test_labels,k)

}

knn_acc<-sapply(knn_res, function(x){x[[2]]})

plot(knn_acc)

knn15_metrics<-confusionMatrix(as.factor(knn_res[[15]][[1]]$pred_class),
                               as.factor(test_labels$label))

# Naive Bayes ------------------------------------------------------------------------------------------------

nb<-train(label~.,
          data = train_full %>% 
            mutate(label = as.factor(label)),
          'nb',
          trControl=trainControl(method='cv',number=5))

nb_pred<-predict(nb,
                 newdata = testset)

# Correlation ---------------------------------------------------------------------------------

correlations<-cor(train_full)

corr_target<-correlations[,4956]

get_index_high_corr<-function(corr,threshold){
  
  high_corr<-list()
  
  for(i in 1:dim(corr)[[1]]){
    for(j in i:dim(corr)[[2]]){
      
      if(corr[[i,j]]>=threshold & i!=j){
      high_corr[[paste(i,j,sep = "_")]]<-c(i,j)
      }
      
    }
  }
  return(high_corr)
}

high_corr<-get_index_high_corr(correlations,0.9)

high_corr_1<-sapply(high_corr, function(x){x[[1]]})[1:20]
high_corr_2<-sapply(high_corr, function(x){x[[2]]})[1:20]


corrplot(cor(train_full %>% 
               dplyr::select(union(high_corr_1,high_corr_2))))
