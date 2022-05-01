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
library(MLmetrics)
library(randomForest)
library(ranger)
library(caret)
library(mlflow)
library(readr)
library(dplyr)
library(caretEnsemble)
library(skimr)
library(igraph)
library(DataExplorer)



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
plot_missing(trainingTables)



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

features<-names(trainingTablesSectionA[c(2:40,43:44,53:58,60:80)])

informativeFeatures <- features
features<-paste(features, collapse="|")


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


mlf_experiment_id = mlflow_set_experiment(
  experiment_name = "RandomForest-loop2"
)



features_p <- mlflow_param("features", features, "string")
mtry <- mlflow_param("mtry", 3, "numeric")
ntree <- mlflow_param("ntree", 3, "numeric")
maxnodes<- mlflow_param("maxnodes", 4, "numeric")


rangerrf <- ranger(yHat ~.,data = train, num.trees = 500, mtry = 3)
MLmetrics::RMSE(testingPreds$predictions,testing$yHat)
rangerrf

#print(predictorrf)
trainPreds      <- predict(rangerrf, train)
validationPreds <- predict(rangerrf, validation)
testingPreds    <- predict(rangerrf, testing)



rmse <- sqrt(mean((testingPreds - test_y) ^ 2))
mae <- mean(abs(testingPreds - test_y))
r2 <- as.numeric(cor(testingPreds, test_y) ^ 2)



#randomForest(yHat ~ ., data = train, mtry = mtry,ntree=ntree,maxnodes=maxnodes,
                          #  importance = TRUE, na.action = na.omit, proximity=TRUE)




  #RandomForest
  for (mtry in 3:4) {
    for (ntree in c(seq(from = 50, to = 600, by  = 50))){
      for (maxnodes in 4:18){
        mlflow_start_run()
        mlflow_log_param("run_name", "randomforest")
        mlflow_log_param("features_p", features)
        mlflow_log_param("mtry", mtry)
        mlflow_log_param("ntree", ntree)
        mlflow_log_param("maxnodes", maxnodes)
        
        rangerrf <- ranger(yHat ~.,data = train, num.trees = ntree, mtry = 3)
        MLmetrics::RMSE(testingPreds$predictions,testing$yHat)
        
        #print(predictorrf)
        $trainPreds      <- predict(predictorrf, train)
        $validationPreds <- predict(predictorrf, validation)
        $testingPreds    <- predict(predictorrf, testing)
        #Evaluating performance on training data
        rmse <- MLmetrics::RMSE(testingPreds$predictions,testing$yHat)
        mae <- MLmetrics::MAE(testingPreds$predictions,testing$yHat)
        r2 <- MLmetrics::R2_Score(testingPreds$predictions,testing$yHat)
        
        mlflow_log_metric("rmse-Train", rmse)
        mlflow_log_metric("r2-Train", r2)
        mlflow_log_metric("mae-Train", mae)
        
        #Evaluating performance on validation data
        rmse <- sqrt(mean((validationPreds - validation_y) ^ 2))
        mae <- mean(abs(validationPreds - validation_y))
        r2 <- as.numeric(cor(validationPreds, validation_y) ^ 2)
        
        mlflow_log_metric("rmse-Validation", rmse)
        mlflow_log_metric("r2-Validation", r2)
        mlflow_log_metric("mae-Validation", mae)
        
        #Evaluating performance on testing data
        rmse <- sqrt(mean((testingPreds - test_y) ^ 2))
        mae <- mean(abs(testingPreds - test_y))
        r2 <- as.numeric(cor(testingPreds, test_y) ^ 2)
        
        mlflow_log_metric("rmse-testing", rmse)
        mlflow_log_metric("r2-testing", r2)
        mlflow_log_metric("mae-testing", mae)
        
        message("Random Forest model (mtry=", 3, ", importance=", "true", "):")
        message("  RMSE: ", rmse)
        message("  MAE: ", mae)
        message("  R2: ", r2)
        #######################################################test###########################
        #mlflow_set_tracking_uri("http://localhost:5712")
        mlflow_end_run()      
      }
    }
    
  }
  
#mlflow_ui()

# Make predictions on the prospect file
#prospectsPreds  <- predict(fitLM, prospects)
#write.csv(prospectsPreds, 'prospectPredictionFile.csv', row.names = F)


# End