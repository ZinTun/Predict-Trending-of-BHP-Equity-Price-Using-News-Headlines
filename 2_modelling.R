# data loading
datatrain=read.csv("train_final.csv")
datatest=read.csv("test_final.csv")

datatrain[1] <- NULL
datatest[1] <- NULL


##################################################
###########       Baseline Model        ##########
##################################################

# Upsampling for data balance
train0 <- datatrain[datatrain$trainy == 0,]
set.seed(123)
index.0 <- sample(1:nrow(train0), (500-nrow(train0)), replace = TRUE)
train0.new <- train0[index.0,]

train1 <- datatrain[datatrain$trainy == 1,]
set.seed(123)
index.1 <- sample(1:nrow(train1), (500-nrow(train1)), replace = TRUE)
train1.new <- train1[index.1,]

datatrain.new <- rbind(datatrain,train0.new,train1.new)

# data processing
trainy=as.factor(datatrain.new[,c("trainy")])
testy=as.factor(datatest[,c("testy")])
trainx_red=datatrain.new[,-c(which(colnames(datatrain.new) %in% c("trainy")))]
testx_red=datatest[,-c(which(colnames(datatest) %in% c("testy")))]

traindf_red=cbind(trainx_red,trainy)
testdf_red=cbind(testx_red,testy)


# calculate accuracy function
calculate_accuracy=function(predy,truey){
        acc=mean(as.numeric(predy == truey))
        return(acc)
}

########## c5.0 ##########
library(C50)
set.seed(123)
mod1=C5.0(trainx_red, trainy)
predtr1=predict(mod1, trainx_red, type = 'class')
predtt1=predict(mod1, testx_red, type = 'class')
calculate_accuracy(predtr1,trainy)
calculate_accuracy(predtt1,testy)
#prune
fitctrl= trainControl(method = "cv", number = 10)
grid2=expand.grid(trials=seq(1:10),model=c("tree","rules"),winnow=c("TRUE","FALSE"))
mod2 = caret::train(trainy~., traindf_red, method = 'C5.0', tuneGrid = grid2,  trControl = fitctrl )
predtr2=predict(mod2, trainx_red, type = 'raw')
predtt2=predict(mod2, testx_red, type = 'raw')
calculate_accuracy(predtr2,trainy)
calculate_accuracy(predtt2,testy)

########## rpart ##########
set.seed(123)
library(rpart)
library(rpart.plot)
mod3=rpart(trainy~., data=traindf_red,control = rpart.control(cp = 0.000001))
predtr3=predict(mod3, trainx_red, type = 'class')
predtt3=predict(mod3, testx_red, type = 'class')
calculate_accuracy(predtr3,trainy)
calculate_accuracy(predtt3,testy)
#prune
bestcp=mod3$cptable[which.min(mod3$cptable[,"xerror"]),"CP"]
bestcp
mod4=prune(mod3, cp=bestcp)
predtr4=predict(mod4, trainx_red, type = 'class')
predtt4=predict(mod4, testx_red, type = 'class')
calculate_accuracy(predtr4,trainy)
calculate_accuracy(predtt4,testy)

########## neural network ##########
set.seed(123)
grid5=expand.grid(size = seq(from = 1, to = 5, by = 1),decay = seq(0,2,0.125))
mod5= caret::train(trainx_red,trainy, method = 'nnet', preProcess = c("center", "scale"), tuneGrid = grid5,  trControl = fitctrl,maxit=500,verbose = FALSE)
predtr5=predict(mod5, trainx_red, type = 'raw')
predtt5=predict(mod5, testx_red, type = 'raw')
calculate_accuracy(predtr5,trainy)
calculate_accuracy(predtt5,testy)

########## SVM ##########
library(e1071)
#prune
tc <- tune.control(cross = 5)
degrees <- c(2, 3, 4, 5, 7)
set.seed(123)
poly_tune <- tune.svm(trainy ~ ., data = traindf_red, degree = degrees,tunecontrol = tc, kernel = "polynomial")
plot(poly_tune)
opt_degree <- poly_tune$best.model$degree
opt_cost <- poly_tune$best.model$cost
set.seed(123)
mod7 <- svm(trainy ~ ., data = traindf_red, traindf_red = opt_degree, type='C-classification',tunecontrol = tc, kernel = "polynomial")
predtr7=predict(mod7, trainx_red, type = 'class')
predtt7=predict(mod7, testx_red, type = 'class')
calculate_accuracy(predtr7,trainy)
calculate_accuracy(predtt7,testy)

gammas <- c(1,2,3,5,6,7,8,9,10)
set.seed(123)
rbf_tune <- tune.svm(trainy ~ ., data = traindf_red, gamma = gammas,tunecontrol = tc, kernel = "radial")
plot(rbf_tune)
opt_gamma <- rbf_tune$best.model$gamma
set.seed(123)
mod8 <- svm(trainy ~ ., data = traindf_red, gamma = opt_gamma, type='C-classification',tunecontrol = tc, kernel = "radial")
predtr8=predict(mod8, trainx_red, type = 'class')
predtt8=predict(mod8, testx_red, type = 'class')
calculate_accuracy(predtr8,trainy)
calculate_accuracy(predtt8,testy)

set.seed(123)
mod9 = svm(trainy ~ ., traindf_red, method = 'class',  kernel = 'sigmoid')
predtr9=predict(mod9, trainx_red, type = 'class')
predtt9=predict(mod9, testx_red, type = 'class')
calculate_accuracy(predtr9,trainy)
calculate_accuracy(predtt9,testy)



##################################################
###########       Ensemble Model        ##########
##################################################

########### Random Forest ##########
fitctrl= trainControl(method = "cv", number = 10)
grid10=expand.grid(mtry=seq(3:15))
mod10 = caret::train(trainy~., traindf_red, method = 'rf', tuneGrid = grid10,  trControl = fitctrl)
predtr10=predict(mod10, trainx_red, type = 'raw')
calculate_accuracy(predtr10,trainy)
predtt10=predict(mod10, testx_red, type = 'raw')
calculate_accuracy(predtt10,testy)

########## XGBoosting ##########
library("xgboost")

# convert data for xgboost
x.train <- model.matrix(trainy ~ ., data=traindf_red)[, -1]
y.train <- as.numeric(traindf_red[, "trainy"]) - 1
dtrain <- xgb.DMatrix(data=x.train, label=y.train)
x.test <- model.matrix(testy ~ ., data=testdf_red)[, -1]
y.test <- as.numeric(testdf_red[, "testy"]) - 1

# tuning xgboost
# input required
# data: dtrain in xgb.DMatrix format
#objective <- "reg:linear"
objective <- "multi:softmax"
cv.fold <- 5

# parameter ranges
max_depths <- c(1, 2, 4, 6)  # candidates for d
etas <- c(0.01, 0.005, 0.001)  # candidates for lambda
subsamples <- c(0.5, 0.75, 1)
colsamples <- c(0.6, 0.8, 1)

set.seed(4321)
tune.out <- data.frame()
for (max_depth in max_depths) {
        for (eta in etas) {
                for (subsample in subsamples) {
                        for (colsample in colsamples) {
                                # **calculate max n.trees by my secret formula**
                                n.max <- round(100 / (eta * sqrt(max_depth)))
                                xgb.cv.fit <- xgb.cv(data = dtrain, objective=objective, nfold=cv.fold, early_stopping_rounds=100, verbose=0,
                                                     nrounds=n.max, num_class= 3, max_depth=max_depth, eta=eta, subsample=subsample, colsample_bytree=colsample)
                                n.best <- xgb.cv.fit$best_ntreelimit
                                if (objective == "reg:linear") {
                                        cv.err <- xgb.cv.fit$evaluation_log$test_rmse_mean[n.best]
                                } else if (objective == "binary:logistic") {
                                        cv.err <- xgb.cv.fit$evaluation_log$test_error_mean[n.best]
                                } else if(objective =="multi:softmax"){
                                        cv.err <- xgb.cv.fit$evaluation_log$test_merror_mea[n.best]
                                }
                                out <- data.frame(max_depth=max_depth, eta=eta, subsample=subsample, colsample=colsample, n.max=n.max, nrounds=n.best, cv.err=cv.err)
                                print(out)
                                tune.out <- rbind(tune.out, out)
                        }
                }
        }
}

tune.out

# parameters after tuning
opt <- which.min(tune.out$cv.err)
max_depth.opt <- tune.out$max_depth[opt]
eta.opt <- tune.out$eta[opt]
subsample.opt <- tune.out$subsample[opt]
colsample.opt <- tune.out$colsample[opt]
nrounds.opt <- tune.out$nrounds[opt]

# fit a boosting model with optimal parameters
set.seed(4321)
xgb_model <- xgboost(data=dtrain, objective=objective, nrounds=nrounds.opt, num_class = 3,max_depth=max_depth.opt, eta=eta.opt, subsample=subsample.opt, colsample_bytree=colsample.opt, verbose=0)

# predict
prob.xgb_train <- predict(xgb_model, x.train)
calculate_accuracy(prob.xgb_train,traindf_red$trainy) 

prob.xgb <- predict(xgb_model, x.test)
calculate_accuracy(prob.xgb,testdf_red$testy) 


########## Bagging ##########
library(rpart)
set.seed(123)
train_index<-sample(1:2018,1800,replace=FALSE)
train_data<-traindf_red[train_index,]
valid_data<-traindf_red[-train_index,]

bagging <- function(n_bags, samples,train_data){
        out = 0
        for(i in 1:n_bags){
                #sample datapoints
                s1 <- sample(1:nrow(train_data), samples, replace = TRUE)
                ti <- train_data[s1,]
                rp_model<-rpart(factor(trainy)~., ti)
                bestcp <- rp_model$cptable[which.min(rp_model$cptable[,"xerror"]),"CP"]
                rp_pruned <- prune(rp_model, cp = bestcp)
                p <- predict(rp_model, valid_data[-37],type='class')
                out_accuracy <-calculate_accuracy(p,factor(valid_data$trainy))
                out=out+out_accuracy
        }
        out <- out/n_bags
        
        return(out)
}

#check rpart performance
accuracy_bagging_rpart<-list()
rpart_bagnum<-list()
rpart_samplenum<-list()
bag_numbers<-c(50,100,200,500)
sample_number<-seq(from=300,to=999,by=100)
index <-1

for (i in bag_numbers) {
        for (j in sample_number){
                bag_q1<-bagging(i,j,train_data)
                accuracy_bagging_rpart[index]<- bag_q1
                rpart_bagnum[index]<-i
                rpart_samplenum[index]<-j
                index<-index+1}
}

mean(unlist(accuracy_bagging_rpart))
max_accuracy_bagging_rpart<- max(as.vector(unlist(accuracy_bagging_rpart)))
max_accuracy_bagging_rpart
best_rpart_bagnum<-as.numeric(rpart_bagnum[which.max(as.vector(unlist(accuracy_bagging_rpart)))])
best_rpart_samplenum<-as.numeric(rpart_samplenum[which.max(as.vector(unlist(accuracy_bagging_rpart)))])

set.seed(123)
s_best<-sample(1:nrow(train_data),best_rpart_samplenum, replace = TRUE)
ti_best <- train_data[s_best,]
rp_model_best<-rpart(factor(trainy)~., ti_best)
bestcp_final <- rp_model_best$cptable[which.min(rp_model_best$cptable[,"xerror"]),"CP"]
rp_best_pruned <- prune(rp_model_best, cp = bestcp_final)
best_bagging_model_pred <- predict(rp_best_pruned, testdf_red[-37],type='class')
best_bagging_test_accuray <-calculate_accuracy(best_bagging_model_pred,factor(testdf_red$testy))
best_bagging_test_accuray



########## Boosting ##########
adaboost_train_data<-train_data
adaboost_train_data$trainy<-factor(adaboost_train_data$trainy)
adaboost_valid_data<-valid_data
adaboost_valid_data$trainy<-factor(adaboost_valid_data$trainy)
adaboost_test_data<-testdf_red
adaboost_test_data$testy<-factor(adaboost_test_data$testy)

accuracy_adaboost<-list()
ada_mfinalnum<-list()
mfinal_num<-seq(from=50,to=500,by=50)
index_ada<-1

#check adaboost performance
library(adabag)
for (m in mfinal_num){
        adaboost <- boosting(trainy~., data=adaboost_train_data, mfinal=m)
        adaboost_pred <- predict(adaboost,adaboost_valid_data)
        adaboost_accuracy<- 1-adaboost_pred$error
        accuracy_adaboost[index_ada]<-adaboost_accuracy
        ada_mfinalnum[index_ada]<-m
        index_ada<-index_ada+1
}

ada_max_accuracy<-max(as.vector(unlist(accuracy_adaboost)))
ada_max_accuracy
ada_mfinal_best<-as.numeric(ada_mfinalnum[which.max(as.vector(unlist(accuracy_adaboost)))])
ada_model_best<-boosting(trainy~., data=adaboost_train_data, mfinal=ada_mfinal_best)
ada_model_best_pred <- predict(ada_model_best,adaboost_test_data)
ada_model_best_pred_accuracy<- calculate_accuracy(ada_model_best_pred$class,factor(adaboost_test_data$testy))
ada_model_best_pred_accuracy



########## Stacking Model ##########
set.seed(123)
stack_index<-sample(1:nrow(traindf_red),1000,replace=FALSE)
stack_traindf<-traindf_red[stack_index,]
stack_validdf<-traindf_red[-stack_index,]
stack_testdf<-testdf_red

# level0- rf
stack_fitctrl= trainControl(method = "cv", number = 10)
stack_grid10=expand.grid(mtry=seq(3:15))
stack_rf_mod10 = caret::train(trainy~.,stack_traindf , method = 'rf', tuneGrid = stack_grid10,  trControl = stack_fitctrl)
stack_level0_rfX=predict(stack_rf_mod10, stack_validdf, type = 'raw')

# level0- svm(RBF)
stack_tc <- tune.control(cross = 5)
stack_svm_gammas <- c(1,2,3,5,6,7,8,9,10)
set.seed(123)
stack_rbf_tune <- tune.svm(trainy ~ ., data = stack_traindf, gamma = stack_svm_gammas,tunecontrol = stack_tc, kernel = "radial")
stack_opt_gamma <- stack_rbf_tune$best.model$gamma
set.seed(123)
stack_rbf_model <- svm(trainy ~ ., data = stack_traindf, gamma = stack_opt_gamma, type='C-classification',tunecontrol = stack_tc, kernel = "radial")
stack_level0_svmX=predict(stack_rbf_model, stack_validdf, type = 'class')

# build data for level 1 model
stack_level0_data<-cbind(stack_level0_rfX,stack_level0_svmX,stack_validdf$trainy)
colnames(stack_level0_data)<-c("stack_level0_rfX","stack_level0_svmX","trainy")

# build level 1 model
# convert data for xgboost
stack_x.train <- model.matrix(trainy ~ ., data=as.data.frame(stack_level0_data))[, -1]
stack_y.train <- as.numeric(stack_level0_data[, "trainy"]) - 1
stack_dtrain <- xgb.DMatrix(data=stack_x.train, label=stack_y.train)
stack_x.test <- model.matrix(testy ~ ., data=stack_testdf)[, -1]
stack_y.test <- as.numeric(stack_testdf[, "testy"]) - 1

# tuning xgboost
stack_objective <- "multi:softmax"
stack_cv.fold <- 5

# parameter ranges
stack_max_depths <- c(1, 2, 4, 6)  # candidates for d
stack_etas <- c(0.01, 0.005, 0.001)  # candidates for lambda
stack_subsamples <- c(0.5, 0.75, 1)
stack_colsamples <- c(0.6, 0.8, 1)

set.seed(4321)
stack_tune.out <- data.frame()
for (max_depth in stack_max_depths) {
        for (eta in stack_etas) {
                for (subsample in stack_subsamples) {
                        for (colsample in stack_colsamples) {
                                # **calculate max n.trees by my secret formula**
                                n.max <- round(100 / (eta * sqrt(max_depth)))
                                xgb.cv.fit <- xgb.cv(data = stack_dtrain, objective=stack_objective, nfold=stack_cv.fold, early_stopping_rounds=100, verbose=0,
                                                     nrounds=n.max, num_class= 3, max_depth=max_depth, eta=eta, subsample=subsample, colsample_bytree=colsample)
                                n.best <- xgb.cv.fit$best_ntreelimit
                                if (stack_objective == "reg:linear") {
                                        cv.err <- xgb.cv.fit$evaluation_log$test_rmse_mean[n.best]
                                } else if (stack_objective == "binary:logistic") {
                                        cv.err <- xgb.cv.fit$evaluation_log$test_error_mean[n.best]
                                } else if(stack_objective =="multi:softmax"){
                                        cv.err <- xgb.cv.fit$evaluation_log$test_merror_mea[n.best]
                                }
                                out <- data.frame(max_depth=max_depth, eta=eta, subsample=subsample, colsample=colsample, n.max=n.max, nrounds=n.best, cv.err=cv.err)
                                print(out)
                                stack_tune.out <- rbind(stack_tune.out, out)
                        }
                }
        }
}

stack_tune.out

# parameters after tuning
stack_opt <- which.min(stack_tune.out$cv.err)
stack_max_depth.opt <- stack_tune.out$max_depth[stack_opt]
stack_eta.opt <- stack_tune.out$eta[stack_opt]
stack_subsample.opt <- stack_tune.out$subsample[stack_opt]
stack_colsample.opt <- stack_tune.out$colsample[stack_opt]
stack_nrounds.opt <- stack_tune.out$nrounds[stack_opt]

# fit a boosting model with optimal parameters
set.seed(4321)
stack_level1_xgb_model <- xgboost(data=stack_dtrain, objective=stack_objective, nrounds=stack_nrounds.opt, num_class = 3,
                                  max_depth=stack_max_depth.opt, eta=stack_eta.opt, subsample=stack_subsample.opt, 
                                  colsample_bytree=stack_colsample.opt, verbose=0)

#use test data to test accuracy
stack_level0_rfX_test=predict(stack_rf_mod10, stack_testdf, type = 'raw')
stack_level0_svmX_test=predict(stack_rbf_model, stack_testdf, type = 'class')
stack_level0_data_test<-cbind(stack_level0_rfX_test,stack_level0_svmX_test,stack_testdf$testy)
colnames(stack_level0_data_test)<-c("stack_level0_rfX_test","stack_level0_svmX_test","testy")
#predict
#test accuracy
stack_x.test_final <- model.matrix(testy ~ ., data=as.data.frame(stack_level0_data_test))[, -1]
stack_prob.xgb_test <- predict(stack_level1_xgb_model, stack_x.test_final)
calculate_accuracy(stack_prob.xgb_test,stack_testdf$testy) 

#use training data to test accuracy
stack_level0_rfX_train=predict(stack_rf_mod10, traindf_red, type = 'raw')
stack_level0_svmX_train=predict(stack_rbf_model, traindf_red, type = 'class')
stack_level0_data_train<-cbind(stack_level0_rfX_train,stack_level0_svmX_train,traindf_red$trainy)
colnames(stack_level0_data_train)<-c("stack_level0_rfX_train","stack_level0_svmX_train","trainy")
#predict
#training accuracy
stack_x.training_final <- model.matrix(trainy ~ ., data=as.data.frame(stack_level0_data_train))[, -1]
stack_prob.xgb_train <- predict(stack_level1_xgb_model, stack_x.training_final)
calculate_accuracy(stack_prob.xgb_train,traindf_red$trainy)




##################################################
###########    With new 3 features      ##########
##################################################

train.newf<-read.csv("data_train.csv")
test.newf<-read.csv("data_test.csv")

train.newf[1] <- NULL
test.newf[1] <- NULL

########## training data processing ##########
# Upsampling for data balance
# df_final, df_final_test/train.newf, test.newf
train0.f <- train.newf[train.newf$ynew == 0,]
train0.new.f <- train0.f[index.0,]

train1.f <- train.newf[train.newf$ynew == 1,]
train1.new.f <- train1.f[index.1,]

datatrain.new.f <- rbind(train.newf,train0.new.f,train1.new.f)


########## modeling with new features ##########
traindf_new <- cbind(traindf_red,datatrain.new.f[4:8])
testdf_new <- cbind(testdf_red,test.newf[4:8])


# data processing
trainy.f=as.factor(traindf_new[,c("trainy")])
testy.f=as.factor(testdf_new[,c("testy")])
trainx_red.f=traindf_new[,-c(which(colnames(traindf_new) %in% c("trainy")))]
testx_red.f=testdf_new[,-c(which(colnames(testdf_new) %in% c("testy")))]

traindf_red.f=cbind(trainx_red.f,trainy.f)
testdf_red.f=cbind(testx_red.f,testy.f)




########## SVM - RBF ##########
tc <- tune.control(cross = 5)
gammas <- c(1,2,3,5,6,7,8,9,10)
set.seed(123)
rbf_tune.f <- tune.svm(trainy.f ~ ., data = traindf_red.f, gamma = gammas,tunecontrol = tc, kernel = "radial")
plot(rbf_tune.f)
opt_gamma.f <- rbf_tune.f$best.model$gamma
set.seed(123)
mod11 <- svm(trainy.f ~ ., data = traindf_red.f, gamma = opt_gamma.f, type='C-classification',tunecontrol = tc, kernel = "radial")
predtr11=predict(mod11, trainx_red.f, type = 'class')
predtt11=predict(mod11, testx_red.f, type = 'class')
calculate_accuracy(predtr11,trainy.f)
calculate_accuracy(predtt11,testy.f)




########## XGBoosting ##########
# convert data for xgboost
x.train.f <- model.matrix(trainy.f ~ ., data=traindf_red.f)[, -1]
y.train.f <- as.numeric(traindf_red.f[, "trainy.f"]) - 1
dtrain.f <- xgb.DMatrix(data=x.train.f, label=y.train.f)
x.test.f <- model.matrix(testy.f ~ ., data=testdf_red.f)[, -1]
y.test.f <- as.numeric(testdf_red.f[, "testy.f"]) - 1

# tuning xgboost
# input required
# data: dtrain in xgb.DMatrix format
#objective <- "reg:linear"
objective <- "multi:softmax"
cv.fold <- 5

# parameter ranges
max_depths <- c(1, 2, 4, 6)  # candidates for d
etas <- c(0.01, 0.005, 0.001)  # candidates for lambda
subsamples <- c(0.5, 0.75, 1)
colsamples <- c(0.6, 0.8, 1)

set.seed(4321)
tune.out.f <- data.frame()
for (max_depth in max_depths) {
        for (eta in etas) {
                for (subsample in subsamples) {
                        for (colsample in colsamples) {
                                # **calculate max n.trees by my secret formula**
                                n.max <- round(100 / (eta * sqrt(max_depth)))
                                xgb.cv.fit <- xgb.cv(data = dtrain.f, objective=objective, nfold=cv.fold, early_stopping_rounds=100, verbose=0,
                                                     nrounds=n.max, num_class= 3, max_depth=max_depth, eta=eta, subsample=subsample, colsample_bytree=colsample)
                                n.best <- xgb.cv.fit$best_ntreelimit
                                if (objective == "reg:linear") {
                                        cv.err <- xgb.cv.fit$evaluation_log$test_rmse_mean[n.best]
                                } else if (objective == "binary:logistic") {
                                        cv.err <- xgb.cv.fit$evaluation_log$test_error_mean[n.best]
                                } else if(objective =="multi:softmax"){
                                        cv.err <- xgb.cv.fit$evaluation_log$test_merror_mea[n.best]
                                }
                                out <- data.frame(max_depth=max_depth, eta=eta, subsample=subsample, colsample=colsample, n.max=n.max, nrounds=n.best, cv.err=cv.err)
                                print(out)
                                tune.out.f <- rbind(tune.out.f, out)
                        }
                }
        }
}

tune.out.f

# parameters after tuning
opt.f <- which.min(tune.out.f$cv.err)
max_depth.opt.f <- tune.out.f$max_depth[opt.f]
eta.opt.f <- tune.out.f$eta[opt.f]
subsample.opt.f <- tune.out.f$subsample[opt.f]
colsample.opt.f <- tune.out.f$colsample[opt.f]
nrounds.opt.f <- tune.out.f$nrounds[opt.f]

# fit a boosting model with optimal parameters
set.seed(4321)
xgb_model.f <- xgboost(data=dtrain.f, objective=objective, nrounds=nrounds.opt.f, num_class = 3,max_depth=max_depth.opt.f, eta=eta.opt.f, subsample=subsample.opt.f, colsample_bytree=colsample.opt.f, verbose=0)

# predict
prob.xgb_train.f <- predict(xgb_model.f, x.train.f)
calculate_accuracy(prob.xgb_train.f,traindf_red.f$trainy.f) 

prob.xgb.f <- predict(xgb_model.f, x.test.f)
calculate_accuracy(prob.xgb.f,testdf_red.f$testy.f) 



