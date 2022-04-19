#' Case III Scaffold
#' Ted Kwartler
#' Apr 11 2022

# Options & Set up
setwd("~/Users/manoj/Harvard Courses/Harvard_DataMining_Business_Student/Cases/III Household Spend/studentTables")
options(scipen=999)

# Libraries
library(plyr)
library(vtreat)
library(ModelMetrics)
library(randomForest)
library(caret)
library(mlflow)
library(readr)
library(dplyr)
library(caretEnsemble)
library(ROSE)
library(SuperLearner)

# Get the files
trainingFiles <- list.files(pattern = '_training')
testingFiles  <- list.files(pattern = '_testing')
prospects     <- list.files(pattern = '_prospects')

# Read & Join the files
trainingTables <- lapply(trainingFiles, read.csv)
trainingTables <- join_all(trainingTables, by = 'tmpID')

testingTables <- lapply(testingFiles, read.csv)
testingTables <- join_all(testingTables, by = 'tmpID')

prospectTables <- lapply(prospects, read.csv)
prospectTables <- join_all(prospectTables, by = 'tmpID')

## Sample
set.seed(1234)
idx <- sample(1:nrow(trainingTables), 12000)
trainingTablesSectionA <- trainingTables[idx,]
trainingTablesSectionB <- trainingTables[-idx,]

glimpse(trainingTables)
summary(trainingTables)
str(trainingTables)

#PLot correlations
################


#####

## Explore -- do more exploration 
names(trainingTablesSectionA)
head(trainingTablesSectionA)
barplot(table(trainingTablesSectionA$NetWorth), las = 2)

## Modify 
# Choose which variables are ethical to use, and others which may not be useful; here I just chose 5 variables

myname<-"manoj"

features<-"4:26,43,44,53:58,60:80"
informativeFeatures <- names(trainingTablesSectionA)[c(4:26,43,44,53:58,60:80)]


plan  <- designTreatmentsN(trainingTablesSectionA, 
                           informativeFeatures,
                           'yHat')

print(informativeFeatures)
# Prepare all sections of the data
train      <- prepare(plan, trainingTablesSectionA)
validation <- prepare(plan, trainingTablesSectionB)
testing    <- prepare(plan, testingTables)
prospects  <- prepare(plan, prospectTables)

## Model(s)


train_x <- as.matrix(train[, !(names(train) == "yHat")])
test_x <- as.matrix(testing[, !(names(train) == "yHat")])
train_y <- train[, "yHat"]
test_y <- testing[, "yHat"]
validation_x <- as.matrix(validation[, !(names(train) == "yHat")])
validation_y <- validation[, "yHat"]

mlflow_set_experiment(
  experiment_name = "Case-III",

)
features <- mlflow_param("features", 0.15, "string")
alpha <- mlflow_param("alpha", 0.15, "numeric")
lambda <- mlflow_param("lambda", 0.45, "numeric")
#mlinformativeFeatures <-mlflow_param("informativeFeatures",as.character(informativeFeatures),"string")


#trying superlearner &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
sampled <- sample(1:nrow(train), 0.15 * nrow(train))
sltrain <- train[sampled, ]
sltest <- train[-sampled, ]
sltrain_x <- sltrain[, !(names(train) == "yHat")]
sltrain_y <- sltrain[, "yHat"]

# Review code for corP, which is based on univariate correlation.
screen.corP

set.seed(1)

# Fit the SuperLearner.
# We need to use list() instead of c().
cv_sl = CV.SuperLearner(Y = sltrain_y, X = sltrain_x, family = gaussian(),
                        # For a real analysis we would use V = 10.
                        cvControl = list(V = 10),
                        parallel = "multicore",
                        SL.library = list("SL.mean", "SL.glmnet","SL.ranger", "SL.xgboost", "SL.gbm","SL.randomForest", c("SL.glmnet", "screen.corP","screen.randomForest")))
summary(cv_sl)


#sl = SuperLearner(Y = sltrain_y, X = sltrain_x, family = gaussian(),parallel = "multicore",
#                  SL.library = c("SL.mean", "SL.glmnet","SL.ranger", "SL.xgboost", "SL.gbm"))
#sl

#trying superlearner&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

with(mlflow_start_run(), {
  
  #######################################################test###########################
  model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "gaussian", standardize = FALSE)
  predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model)
  predicted <- predictor(test_x)
  
  rmse <- sqrt(mean((predicted - test_y) ^ 2))
  mae <- mean(abs(predicted - test_y))
  r2 <- as.numeric(cor(predicted, test_y) ^ 2)
  
  message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
  message("  RMSE: ", rmse)
  message("  MAE: ", mae)
  message("  R2: ", r2)
  
  #mlflow_log_param("informativeFeatures", mlinformativeFeatures)
  
  mlflow_log_param("alpha", alpha)
  mlflow_log_param("lambda", lambda)
  mlflow_log_metric("rmse-glmnet", rmse)
  mlflow_log_metric("r2-glmnet", r2)
  mlflow_log_metric("mae-glmnet", mae)
  
  mlflow_log_model(predictor, "model")
  
  #lm
  fitLM <- lm(yHat ~ ., train)
  
  ## Assess all models
  summary(fitLM)
  
  trainPreds      <- predict(fitLM, train)
  validationPreds <- predict(fitLM, validation)
  testingPreds    <- predict(fitLM, testing)
  
  # Choose the best model; evaluate all models and look for consistency among training, validation and test sets.  You dont know the propsect yHat so you can't evaluate that.
  rmse(train$yHat, trainPreds)
  
  
  #Evaluating performance on training data
  rmse <- sqrt(mean((trainPreds - train_y) ^ 2))
  mae <- mean(abs(trainPreds - train_y))
  r2 <- as.numeric(cor(trainPreds, train_y) ^ 2)
  
  mlflow_log_metric("rmse-lm - Train", rmse)
  mlflow_log_metric("r2-lm - Train", r2)
  mlflow_log_metric("mae-lm - Train", mae)
  
  #Evaluating performance on validation data
  rmse <- sqrt(mean((validationPreds - validation_y) ^ 2))
  mae <- mean(abs(validationPreds - validation_y))
  r2 <- as.numeric(cor(validationPreds, validation_y) ^ 2)
  
  mlflow_log_metric("rmse-lm - Validation", rmse)
  mlflow_log_metric("r2-lm - Validation", r2)
  mlflow_log_metric("mae-lm - Validation", mae)
  
  #Evaluating performance on testing data
  rmse <- sqrt(mean((testingPreds - test_y) ^ 2))
  mae <- mean(abs(testingPreds - test_y))
  r2 <- as.numeric(cor(testingPreds, test_y) ^ 2)
  
  mlflow_log_metric("rmse-lm - testing", rmse)
  mlflow_log_metric("r2-lm - testing", r2)
  mlflow_log_metric("mae-lm - testing", mae)
  
  #RandomForest
  predictorrf <- randomForest(yHat ~ ., data = train, mtry = 3,
                              importance = TRUE, na.action = na.omit)
  
  print(predictorrf)
  trainPreds      <- predict(predictorrf, train)
  validationPreds <- predict(predictorrf, validation)
  testingPreds    <- predict(predictorrf, testing)
  
  #Evaluating performance on training data
  rmse <- sqrt(mean((trainPreds - train_y) ^ 2))
  mae <- mean(abs(trainPreds - train_y))
  r2 <- as.numeric(cor(trainPreds, train_y) ^ 2)
  
  mlflow_log_metric("rmse-RF - Train", rmse)
  mlflow_log_metric("r2-RF - Train", r2)
  mlflow_log_metric("mae-RF - Train", mae)
  
  #Evaluating performance on validation data
  rmse <- sqrt(mean((validationPreds - validation_y) ^ 2))
  mae <- mean(abs(validationPreds - validation_y))
  r2 <- as.numeric(cor(validationPreds, validation_y) ^ 2)
  
  mlflow_log_metric("rmse-RF - Validation", rmse)
  mlflow_log_metric("r2-RF  - Validation", r2)
  mlflow_log_metric("mae-RF - Validation", mae)
  
  #Evaluating performance on testing data
  rmse <- sqrt(mean((testingPreds - test_y) ^ 2))
  mae <- mean(abs(testingPreds - test_y))
  r2 <- as.numeric(cor(testingPreds, test_y) ^ 2)
  
  message("Random Forest model (mtry=", 3, ", importance=", "true", "):")
  message("  RMSE: ", rmse)
  message("  MAE: ", mae)
  message("  R2: ", r2)
  
  mlflow_log_metric("rmse-RF - testing", rmse)
  mlflow_log_metric("r2-RF  - testing", r2)
  mlflow_log_metric("mae-RF - testing", mae)
  #######################################################test###########################
})

mlflow_set_tracking_uri("http://localhost:5733")
mlflow_end_run()
#mlflow_ui()

# Make predictions on the prospect file
#prospectsPreds  <- predict(fitLM, prospects)
#write.csv(prospectsPreds, 'prospectPredictionFile.csv', row.names = F)


# End