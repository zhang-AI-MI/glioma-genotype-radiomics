library(brnn)
library(pROC)
library(caret)
######
# function: predict genotype based on the selected features using brnn
######

# cross validation for selection of the parameters in the brnn model
cv.brnn<-function(x, y, cv=10, epochs=seq(100,1000,100), neurons=1:5, mus=seq(0.001,0.01,0.001)){
  
  folds <- createFolds(factor(y), k=cv, list=FALSE)
  best_auc = 0
  best_epoch = 0
  best_neuron = 0
  best_mu = 0
  for (epoch in epochs){
    for (neuron in neurons) {
      for (mu in mus) {
        
        auc_cv = array(0)
        for (i in 1:cv){
          x_train <- x[folds!=i,]
          x_test  <- x[folds==i,]
          y_train <- y[folds!=i]
          y_test <- y[folds==i]
          model <- brnn(as.matrix(x_train), y_train, neurons = neuron, epochs=epoch, mu=mu, normalize = FALSE, verbose=FALSE)
          auc_cv[i] <- roc(y_test, predict.brnn(model, x_test))$auc
        }
        
        if (mean(auc_cv) > best_auc){
          best_auc = mean(auc_cv)
          best_auc_cv = auc_cv
          best_epoch = epoch
          best_neuron = neuron
          best_mu = mu
        }
      }
    }
  }
  return (c(best_auc=best_auc, best_auc_cv=best_auc_cv, best_epoch=best_epoch, best_neuron=best_neuron, best_mu=best_mu))
}

# leave-one-out cross validation
loocv.brnn<-function(x, y, epochs=seq(100,1000,100), neurons=1:5, mus=seq(0.001,0.01,0.001)){
  
  folds <- createFolds(factor(y), k=238, list=FALSE)
  best_auc = 0
  best_epoch = 0
  best_neuron = 0
  best_mu = 0
  for (epoch in epochs){
    for (neuron in neurons) {
      for (mu in mus) {
        
        # auc_cv = array(0)
        score_ = array(0)
        for (i in 1:238){
          x_train <- x[folds!=i,]
          x_test  <- x[folds==i,]
          y_train <- y[folds!=i]
          y_test <- y[folds==i]
          model <- brnn(as.matrix(x_train), y_train, neurons = neuron, epochs=epoch, mu=mu, normalize = FALSE, verbose=FALSE)
          score_[i] <- predict.brnn(model, x_test)
        }
        auc_ <- roc(y, score_)$auc
        
        if (auc_ > best_auc){
          best_auc = auc_
          best_auc_cv = auc_
          best_epoch = epoch
          best_neuron = neuron
          best_mu = mu
        }
      }
    }
  }
  return (c(best_auc=best_auc, best_auc_cv=best_auc_cv, best_epoch=best_epoch, best_neuron=best_neuron, best_mu=best_mu))
}

path = 'xxx'#your datapath

info_train <- read.csv(paste(path, 'info_train.csv', sep = ''),header = T)
info_test  <- read.csv(paste(path, 'info_test.csv', sep = ''),header = T)

#######################################IDH##########################################
idh_train <- info_train$IDHmutation
idh_test  <- info_test$IDHmutation

# your direction to the selected features file.
feature_t1_train  <- read.csv(paste(path, 'feature_selection/idh_t1_train1.csv', sep = ''),header = T)
feature_t1_test   <- read.csv(paste(path, 'feature_selection/idh_t1_test1.csv', sep = ''),header = T)
feature_t1c_train <- read.csv(paste(path, 'feature_selection/idh_t1c_train1.csv', sep = ''),header = T)
feature_t1c_test  <- read.csv(paste(path, 'feature_selection/idh_t1c_test1.csv', sep = ''),header = T)
feature_t2_train  <- read.csv(paste(path, 'feature_selection/idh_t2_train1.csv', sep = ''),header = T)
feature_t2_test   <- read.csv(paste(path, 'feature_selection/idh_t2_test1.csv', sep = ''),header = T)
feature_flr_train <- read.csv(paste(path, 'feature_selection/idh_flair_train1.csv', sep = ''),header = T)
feature_flr_test  <- read.csv(paste(path, 'feature_selection/idh_flair_test1.csv', sep = ''),header = T)
feature_dw1_train <- read.csv(paste(path, 'feature_selection/idh_dwi1_train.csv', sep = ''),header = T)
feature_dw1_test  <- read.csv(paste(path, 'feature_selection/idh_dwi1_test.csv', sep = ''),header = T)
feature_dw2_train <- read.csv(paste(path, 'feature_selection/idh_dwi2_train.csv', sep = ''),header = T)
feature_dw2_test  <- read.csv(paste(path, 'feature_selection/idh_dwi2_test.csv', sep = ''),header = T)
feature_adc_train <- read.csv(paste(path, 'feature_selection/idh_adc_train1.csv', sep = ''),header = T)
feature_adc_test  <- read.csv(paste(path, 'feature_selection/idh_adc_test1.csv', sep = ''),header = T)

names(feature_t1_train) <- paste(names(feature_t1_train), '_T1', sep = '')
names(feature_t1_test)  <- paste(names(feature_t1_test), '_T1', sep = '')
names(feature_t1c_train) <- paste(names(feature_t1c_train), '_T1C', sep = '')
names(feature_t1c_test)  <- paste(names(feature_t1c_test), '_T1C', sep = '')
names(feature_t2_train) <- paste(names(feature_t2_train), '_T2', sep = '')
names(feature_t2_test)  <- paste(names(feature_t2_test), '_T2', sep = '')
names(feature_flr_train) <- paste(names(feature_flr_train), '_FLR', sep = '')
names(feature_flr_test)  <- paste(names(feature_flr_test), '_FLR', sep = '')
names(feature_dw1_train) <- paste(names(feature_dw1_train), '_DW1', sep = '')
names(feature_dw1_test)  <- paste(names(feature_dw1_test), '_DW1', sep = '')
names(feature_dw2_train) <- paste(names(feature_dw2_train), '_DW2', sep = '')
names(feature_dw2_test)  <- paste(names(feature_dw2_test), '_DW2', sep = '')
names(feature_adc_train) <- paste(names(feature_adc_train), '_ADC', sep = '')
names(feature_adc_test)  <- paste(names(feature_adc_test), '_ADC', sep = '')

result <- cv.brnn(as.matrix(feature_t1_train), idh_train, cv=10, epochs=seq(100,500,50))
model_idh_t1 <- brnn(as.matrix(feature_t1_train), idh_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_idh_t1_train <- predict.brnn(model_idh_t1, newdata=as.matrix(feature_t1_train))
pred_idh_t1_test  <- predict.brnn(model_idh_t1, newdata=as.matrix(feature_t1_test))
roc(idh_train, pred_idh_t1_train)
roc(idh_test, pred_idh_t1_test)

result <- cv.brnn(as.matrix(feature_t1c_train), idh_train, cv=10, epochs = seq(50,500,50))
model_idh_t1c <- brnn(as.matrix(feature_t1c_train), idh_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_idh_t1c_train <- predict.brnn(model_idh_t1c, newdata=as.matrix(feature_t1c_train))
pred_idh_t1c_test  <- predict.brnn(model_idh_t1c, newdata=as.matrix(feature_t1c_test))
roc(idh_train, pred_idh_t1c_train)
roc(idh_test, pred_idh_t1c_test) 

result <- cv.brnn(as.matrix(feature_t2_train), idh_train, cv=10, epochs = seq(50,500,50))
model_idh_t2 <- brnn(as.matrix(feature_t2_train), idh_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_idh_t2_train <- predict.brnn(model_idh_t2, newdata=as.matrix(feature_t2_train))
pred_idh_t2_test  <- predict.brnn(model_idh_t2, newdata=as.matrix(feature_t2_test))
roc(idh_train, pred_idh_t2_train)
roc(idh_test, pred_idh_t2_test)

result <- cv.brnn(as.matrix(feature_flr_train), idh_train, cv=10, epochs = seq(50,500,50))
model_idh_flr <- brnn(as.matrix(feature_flr_train), idh_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_idh_flr_train <- predict.brnn(model_idh_flr, newdata=as.matrix(feature_flr_train))
pred_idh_flr_test  <- predict.brnn(model_idh_flr, newdata=as.matrix(feature_flr_test))
roc(idh_train, pred_idh_flr_train)
roc(idh_test, pred_idh_flr_test)

result <- cv.brnn(as.matrix(feature_adc_train), idh_train, cv=10, epochs = seq(50,500,50))
model_idh_adc <- brnn(as.matrix(feature_adc_train), idh_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_idh_adc_train <- predict.brnn(model_idh_adc, newdata=as.matrix(feature_adc_train))
pred_idh_adc_test  <- predict.brnn(model_idh_adc, newdata=as.matrix(feature_adc_test))
roc(idh_train, pred_idh_adc_train)
roc(idh_test, pred_idh_adc_test)

# fusion
predscore_train_pd <- data.frame(class=idh_train, t1=pred_idh_t1_train, t1c=pred_idh_t1c_train, t2=pred_idh_t2_train, flr=pred_idh_flr_train,  adc=pred_idh_adc_train)
predscore_test_pd  <- data.frame(class=idh_test, t1=pred_idh_t1_test, t1c=pred_idh_t1c_test, t2=pred_idh_t2_test, flr=pred_idh_flr_test, adc=pred_idh_adc_test)

imgfusion_idh <- lm(class~ ., data = predscore_train_pd)
library(stats)
imgfusion_idh <- step(imgfusion_idh, direction = "both", n=log(238))

pred_idh_train <- predict(imgfusion_idh, newdata = predscore_train_pd)
pred_idh_test  <- predict(imgfusion_idh, newdata = predscore_test_pd)

roc(predscore_train_pd$class, pred_idh_train) 
roc(predscore_test_pd$class, pred_idh_test) 

#######################################1p19q##########################################
pq_train <- info_train$X1p19q.codeletion
pq_test  <- info_test$X1p19q.codeletion

feature_t1_train  <- read.csv(paste(path, 'feature_selection/pq_t1_train1.csv', sep = ''),header = T)
feature_t1_test   <- read.csv(paste(path, 'feature_selection/pq_t1_test1.csv', sep = ''),header = T)
feature_t1c_train <- read.csv(paste(path, 'feature_selection/pq_t1c_train1.csv', sep = ''),header = T)
feature_t1c_test  <- read.csv(paste(path, 'feature_selection/pq_t1c_test1.csv', sep = ''),header = T)
feature_t2_train  <- read.csv(paste(path, 'feature_selection/pq_t2_train1.csv', sep = ''),header = T)
feature_t2_test   <- read.csv(paste(path, 'feature_selection/pq_t2_test1.csv', sep = ''),header = T)
feature_flr_train <- read.csv(paste(path, 'feature_selection/pq_flair_train1.csv', sep = ''),header = T)
feature_flr_test  <- read.csv(paste(path, 'feature_selection/pq_flair_test1.csv', sep = ''),header = T)
feature_dw1_train <- read.csv(paste(path, 'feature_selection/pq_dwi1_train.csv', sep = ''),header = T)
feature_dw1_test  <- read.csv(paste(path, 'feature_selection/pq_dwi1_test.csv', sep = ''),header = T)
feature_dw2_train <- read.csv(paste(path, 'feature_selection/pq_dwi2_train.csv', sep = ''),header = T)
feature_dw2_test  <- read.csv(paste(path, 'feature_selection/pq_dwi2_test.csv', sep = ''),header = T)
feature_adc_train <- read.csv(paste(path, 'feature_selection/pq_adc_train1.csv', sep = ''),header = T)
feature_adc_test  <- read.csv(paste(path, 'feature_selection/pq_adc_test1.csv', sep = ''),header = T)

names(feature_t1_train) <- paste(names(feature_t1_train), '_T1', sep = '')
names(feature_t1_test)  <- paste(names(feature_t1_test), '_T1', sep = '')
names(feature_t1c_train) <- paste(names(feature_t1c_train), '_T1C', sep = '')
names(feature_t1c_test)  <- paste(names(feature_t1c_test), '_T1C', sep = '')
names(feature_t2_train) <- paste(names(feature_t2_train), '_T2', sep = '')
names(feature_t2_test)  <- paste(names(feature_t2_test), '_T2', sep = '')
names(feature_flr_train) <- paste(names(feature_flr_train), '_FLR', sep = '')
names(feature_flr_test)  <- paste(names(feature_flr_test), '_FLR', sep = '')
names(feature_dw1_train) <- paste(names(feature_dw1_train), '_DW1', sep = '')
names(feature_dw1_test)  <- paste(names(feature_dw1_test), '_DW1', sep = '')
names(feature_dw2_train) <- paste(names(feature_dw2_train), '_DW2', sep = '')
names(feature_dw2_test)  <- paste(names(feature_dw2_test), '_DW2', sep = '')
names(feature_adc_train) <- paste(names(feature_adc_train), '_ADC', sep = '')
names(feature_adc_test)  <- paste(names(feature_adc_test), '_ADC', sep = '')

result <- cv.brnn(as.matrix(feature_t1_train), pq_train, cv=10)
model_pq_t1 <- brnn(as.matrix(feature_t1_train), pq_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_pq_t1_train <- predict.brnn(model_pq_t1, newdata=as.matrix(feature_t1_train))
pred_pq_t1_test  <- predict.brnn(model_pq_t1, newdata=as.matrix(feature_t1_test))
roc(pq_train, pred_pq_t1_train)
roc(pq_test, pred_pq_t1_test) 

result <- cv.brnn(as.matrix(feature_t1c_train), pq_train, cv=10)
model_pq_t1c <- brnn(as.matrix(feature_t1c_train), pq_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_pq_t1c_train <- predict.brnn(model_pq_t1c, newdata=as.matrix(feature_t1c_train))
pred_pq_t1c_test  <- predict.brnn(model_pq_t1c, newdata=as.matrix(feature_t1c_test))
roc(pq_train, pred_pq_t1c_train)
roc(pq_test, pred_pq_t1c_test)  

result <- cv.brnn(as.matrix(feature_t2_train), pq_train, cv=10)
model_pq_t2 <- brnn(as.matrix(feature_t2_train), pq_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_pq_t2_train <- predict.brnn(model_pq_t2, newdata=as.matrix(feature_t2_train))
pred_pq_t2_test  <- predict.brnn(model_pq_t2, newdata=as.matrix(feature_t2_test))
roc(pq_train, pred_pq_t2_train)
roc(pq_test, pred_pq_t2_test)

result <- cv.brnn(as.matrix(feature_flr_train), pq_train, cv=10)
model_pq_flr <- brnn(as.matrix(feature_flr_train), pq_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_pq_flr_train <- predict.brnn(model_pq_flr, newdata=as.matrix(feature_flr_train))
pred_pq_flr_test  <- predict.brnn(model_pq_flr, newdata=as.matrix(feature_flr_test))
roc(pq_train, pred_pq_flr_train)
roc(pq_test, pred_pq_flr_test) 

result <- cv.brnn(as.matrix(feature_adc_train), pq_train, cv=10)
model_pq_adc <- brnn(as.matrix(feature_adc_train), pq_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_pq_adc_train <- predict.brnn(model_pq_adc, newdata=as.matrix(feature_adc_train))
pred_pq_adc_test  <- predict.brnn(model_pq_adc, newdata=as.matrix(feature_adc_test))
roc(pq_train, pred_pq_adc_train) 
roc(pq_test, pred_pq_adc_test)

# 1p19q fusion
predscore_train_pd <- data.frame(class=pq_train, t1=pred_pq_t1_train, t1c=pred_pq_t1c_train, t2=pred_pq_t2_train, flr=pred_pq_flr_train, adc=pred_pq_adc_train)
predscore_test_pd  <- data.frame(class=pq_test, t1=pred_pq_t1_test, t1c=pred_pq_t1c_test, t2=pred_pq_t2_test, flr=pred_pq_flr_test, adc=pred_pq_adc_test)

imgfusion_pq <- lm(class~., data = predscore_train_pd)
library(stats)
imgfusion_pq <- step(imgfusion_pq, direction = "both", n=log(238))

pred_pq_train <- predict(imgfusion_pq, newdata = predscore_train_pd)
pred_pq_test  <- predict(imgfusion_pq, newdata = predscore_test_pd)

roc(predscore_train_pd$class, pred_pq_train)
roc(predscore_test_pd$class, pred_pq_test) 

#######################################tert##########################################
tert_train <- info_train$TERTmutation
tert_test  <- info_test$TERTmutation

feature_t1_train  <- read.csv(paste(path, 'feature_selection/tert_t1_train1.csv', sep = ''),header = T)
feature_t1_test   <- read.csv(paste(path, 'feature_selection/tert_t1_test1.csv', sep = ''),header = T)
feature_t1c_train <- read.csv(paste(path, 'feature_selection/tert_t1c_train1.csv', sep = ''),header = T)
feature_t1c_test  <- read.csv(paste(path, 'feature_selection/tert_t1c_test1.csv', sep = ''),header = T)
feature_t2_train  <- read.csv(paste(path, 'feature_selection/tert_t2_train1.csv', sep = ''),header = T)
feature_t2_test   <- read.csv(paste(path, 'feature_selection/tert_t2_test1.csv', sep = ''),header = T)
feature_flr_train <- read.csv(paste(path, 'feature_selection/tert_flair_train1.csv', sep = ''),header = T)
feature_flr_test  <- read.csv(paste(path, 'feature_selection/tert_flair_test1.csv', sep = ''),header = T)
feature_dw1_train <- read.csv(paste(path, 'feature_selection/tert_dwi1_train.csv', sep = ''),header = T)
feature_dw1_test  <- read.csv(paste(path, 'feature_selection/tert_dwi1_test.csv', sep = ''),header = T)
feature_dw2_train <- read.csv(paste(path, 'feature_selection/tert_dwi2_train.csv', sep = ''),header = T)
feature_dw2_test  <- read.csv(paste(path, 'feature_selection/tert_dwi2_test.csv', sep = ''),header = T)
feature_adc_train <- read.csv(paste(path, 'feature_selection/tert_adc_train1.csv', sep = ''),header = T)
feature_adc_test  <- read.csv(paste(path, 'feature_selection/tert_adc_test1.csv', sep = ''),header = T)

names(feature_t1_train) <- paste(names(feature_t1_train), '_T1', sep = '')
names(feature_t1_test)  <- paste(names(feature_t1_test), '_T1', sep = '')
names(feature_t1c_train) <- paste(names(feature_t1c_train), '_T1C', sep = '')
names(feature_t1c_test)  <- paste(names(feature_t1c_test), '_T1C', sep = '')
names(feature_t2_train) <- paste(names(feature_t2_train), '_T2', sep = '')
names(feature_t2_test)  <- paste(names(feature_t2_test), '_T2', sep = '')
names(feature_flr_train) <- paste(names(feature_flr_train), '_FLR', sep = '')
names(feature_flr_test)  <- paste(names(feature_flr_test), '_FLR', sep = '')
names(feature_dw1_train) <- paste(names(feature_dw1_train), '_DW1', sep = '')
names(feature_dw1_test)  <- paste(names(feature_dw1_test), '_DW1', sep = '')
names(feature_dw2_train) <- paste(names(feature_dw2_train), '_DW2', sep = '')
names(feature_dw2_test)  <- paste(names(feature_dw2_test), '_DW2', sep = '')
names(feature_adc_train) <- paste(names(feature_adc_train), '_ADC', sep = '')
names(feature_adc_test)  <- paste(names(feature_adc_test), '_ADC', sep = '')

result <- cv.brnn(as.matrix(feature_t1_train), tert_train, cv=10, epochs = seq(50,200,5), neurons = 1)
model_tert_t1 <- brnn(as.matrix(feature_t1_train), tert_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_tert_t1_train <- predict.brnn(model_tert_t1, newdata=as.matrix(feature_t1_train))
pred_tert_t1_test  <- predict.brnn(model_tert_t1, newdata=as.matrix(feature_t1_test))
roc(tert_train, pred_tert_t1_train)
roc(tert_test, pred_tert_t1_test)  

result <- cv.brnn(as.matrix(feature_t1c_train), tert_train, cv=10, neurons=1)
model_tert_t1c <- brnn(as.matrix(feature_t1c_train), tert_train, neurons=result['best_neuron'], epochs=result['best_epoch'],mu=result['best_mu'], normalize = FALSE)
pred_tert_t1c_train <- predict.brnn(model_tert_t1c, newdata=as.matrix(feature_t1c_train))
pred_tert_t1c_test  <- predict.brnn(model_tert_t1c, newdata=as.matrix(feature_t1c_test))
roc(tert_train, pred_tert_t1c_train)
roc(tert_test, pred_tert_t1c_test)

result <- cv.brnn(as.matrix(feature_t2_train), tert_train, cv=10, neurons=1)
model_tert_t2 <- brnn(as.matrix(feature_t2_train), tert_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_tert_t2_train <- predict.brnn(model_tert_t2, newdata=as.matrix(feature_t2_train))
pred_tert_t2_test  <- predict.brnn(model_tert_t2, newdata=as.matrix(feature_t2_test))
roc(tert_train, pred_tert_t2_train) 
roc(tert_test, pred_tert_t2_test) 

result <- cv.brnn(as.matrix(feature_flr_train),tert_train, cv=10, neurons=1)
model_tert_flr <- brnn(as.matrix(feature_flr_train), tert_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_tert_flr_train <- predict.brnn(model_tert_flr, newdata=as.matrix(feature_flr_train))
pred_tert_flr_test  <- predict.brnn(model_tert_flr, newdata=as.matrix(feature_flr_test))
roc(tert_train, pred_tert_flr_train)
roc(tert_test, pred_tert_flr_test) 

result <- cv.brnn(as.matrix(feature_adc_train), tert_train, cv=10, neurons = 1)
model_tert_adc <- brnn(as.matrix(feature_adc_train), tert_train, neurons=result['best_neuron'], epochs=result['best_epoch'], mu=result['best_mu'], normalize = FALSE)
pred_tert_adc_train <- predict.brnn(model_tert_adc, newdata=as.matrix(feature_adc_train))
pred_tert_adc_test  <- predict.brnn(model_tert_adc, newdata=as.matrix(feature_adc_test))
roc(tert_train, pred_tert_adc_train)
roc(tert_test, pred_tert_adc_test)

# tert fusion
predscore_train_pd <- data.frame(class=tert_train, t1=pred_tert_t1_train, t1c=pred_tert_t1c_train, t2=pred_tert_t2_train, flr=pred_tert_flr_train, adc=pred_tert_adc_train)
predscore_test_pd  <- data.frame(class=tert_test, t1=pred_tert_t1_test, t1c=pred_tert_t1c_test, t2=pred_tert_t2_test, flr=pred_tert_flr_test, adc=pred_tert_adc_test)

imgfusion_tert <- lm(class~t1c+adc, data = predscore_train_pd)
library(stats)
imgfusion_tert <- step(imgfusion_tert, direction = "both", n=log(238))

pred_tert_train <- predict(imgfusion_tert, newdata = predscore_train_pd)
pred_tert_test  <- predict(imgfusion_tert, newdata = predscore_test_pd)

roc(predscore_train_pd$class, pred_tert_train)
roc(predscore_test_pd$class, pred_tert_test) 


