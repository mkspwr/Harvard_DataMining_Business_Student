library(data.table)
library(Matrix)
library(lightgbm)  # library(xgboost)
library(Metrics)

setwd("C:/Users/mksharma/Harvard/Harvard_DataMining_Business_Student")
setwd("./Cases/II National City Bank/training")

TARGET = "Y_AcceptedOffer"

TRAIN_FILE = "training.csv"
TEST_FILE = "testing.csv"


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)



y_train = log(train[,TARGET, with = FALSE])[[TARGET]]
y_test = log(test[,TARGET, with = FALSE])[[TARGET]]

train[, c("V1", TARGET) := NULL]
test[, c("V1") := NULL]

head(train)
dim(train)

lgb.dy1 = lgb.Dataset(data=as.matrix(train), label=y_train)

train_params <- list(
  num_leaves = 4L
  , learning_rate = 1.0
  , objective = "binary"
  , nthread = 2L
)

lgb1d <- lightgbm(
  boosting_type = 'gbdt', 
  objective = "regression", 
  metric = 'mae', 
  lgb.dy1, 
  nrounds = 5000
  ) 

lgbpred12d <- predict(lgb1d, dat = as.matrix(train))
predictions<-as.data.frame(lgbpred12d)
predictions$accepted<-ifelse(predictions$lgbpred12d >= 0, 1,0)
names(training)

training<-read.csv("training.csv")
names(training)

table(training$Y_AcceptedOffer)
table(predictions$accepted)



