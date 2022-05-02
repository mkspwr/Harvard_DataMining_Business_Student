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
#adding a cardinal variable for bookbuyer
#training
trainingTablesSectionA$bookbuyerlevel <- gsub(' book purchase in home', '', trainingTablesSectionA$BookBuyerInHome)
trainingTablesSectionA$bookbuyerlevel <- gsub('book purchases in home', '', trainingTablesSectionA$bookbuyerlevel)
trainingTablesSectionA$bookbuyerlevel[trainingTablesSectionA$bookbuyerlevel == ""] <- "0"
trainingTablesSectionA$bookbuyerlevel<-as.numeric(trainingTablesSectionA$bookbuyerlevel)
#validation
trainingTablesSectionB$bookbuyerlevel <- gsub(' book purchase in home', '', trainingTablesSectionB$BookBuyerInHome)
trainingTablesSectionB$bookbuyerlevel <- gsub('book purchases in home', '', trainingTablesSectionB$bookbuyerlevel)
trainingTablesSectionB$bookbuyerlevel[trainingTablesSectionB$bookbuyerlevel == ""] <- "0"
trainingTablesSectionB$bookbuyerlevel<-as.numeric(trainingTablesSectionB$bookbuyerlevel)
#testing
testingTables$bookbuyerlevel <- gsub(' book purchase in home', '', testingTables$BookBuyerInHome)
testingTables$bookbuyerlevel <- gsub('book purchases in home', '', testingTables$bookbuyerlevel)
testingTables$bookbuyerlevel[testingTables$bookbuyerlevel == ""] <- "0"
testingTables$bookbuyerlevel<-as.numeric(testingTables$bookbuyerlevel)
#Prospect

prospectTables$bookbuyerlevel <- gsub(' book purchase in home', '', prospectTables$BookBuyerInHome)
prospectTables$bookbuyerlevel <- gsub('book purchases in home', '', prospectTables$bookbuyerlevel)
prospectTables$bookbuyerlevel[prospectTables$bookbuyerlevel == ""] <- "0"
prospectTables$bookbuyerlevel<-as.numeric(prospectTables$bookbuyerlevel)


#features<-names(trainingTablesSectionA[c(2:40,43:44,53:58,60:80)])
features<-c('PartiesDescription','storeVisitFrequency','state',
            'city',
            'county',
            'ResidenceHHGenderDescription',
            'Gender',
            'Age',
            'EstHomeValue',
            'MedianEducationYears',
            'ISPSA',
            'PropertyType',
            'HomeOwnerRenter',
            'BroadEthnicGroupings',
            'EthnicDescription',
            'PresenceOfChildrenCode',
            'DwellingUnitSize',
            'Education',
            'NetWorth',
            'ComputerOwnerInHome',
            'LandValue',
            'ReligionsDescription',
            'DonatesToCharityInHome',
            'Investor',
            'OccupationIndustry',
            'bookbuyerlevel',
            'DonatestoLocalCommunity',
            'HealthFitnessMagazineInHome',
            'GeneralCollectorInHousehold',
            'PoliticalContributerInHome',
            'FamilyMagazineInHome',
            'DonatestoHealthcare1',
            'InterestinCurrentAffairsPoliticsInHousehold',
            'MosaicZ4',
            'BuyerofArtinHousehold',
            'GunOwner',
            'HomeOffice',
            'OtherPetOwner',
            'DogOwner',
            'FinancialMagazineInHome',
            'ReligiousContributorInHome',
            'DonatestoWildlifePreservation',
            'DoItYourselfMagazineInHome',
            'CatOwner',
            'LikelyUnionMember',
            'DonatestoChildrensCauses',
            'GardeningMagazineInHome',
            'DonatestoAnimalWelfare',
            'DonatesEnvironmentCauseInHome',
            'FemaleOrientedMagazineInHome',
            'Veteran',
            'CulinaryInterestMagazineInHome',
            'DonatestoVeteransCauses',
            'UpscaleBuyerInHome')

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
  experiment_name = "Case-III Run May5#2"
)



features_p <- mlflow_param("features", features, "string")
alpha <- mlflow_param("alpha", 0.15, "numeric")
lambda <- mlflow_param("lambda", 0.45, "numeric")
run_name<-mlflow_param("name","model_name","string")


#trying caretEnsemble &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# sampled <- sample(1:nrow(train), 0.15 * nrow(train))
# sltrain <- train[sampled, ]
# sltest <- train[-sampled, ]
# sltrain_x <- sltrain[, !(names(train) == "yHat")]
# sltrain_y <- sltrain[, "yHat"]
# 
# 
# # NOT RUN {
# set.seed(42)
# models <- caretList(iris[1:50,1:2], iris[1:50,3], methodList=c("glm", "lm"))
# ens <- caretEnsemble(models)
# summary(ens)
# # }
# 
# obj <- featurePlot(x=trainingTables[,c("Gender")],
#                    y = trainingTables$yHat,
#                    plot="box")
# plot(obj)

#trying caretEnsemble &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

mlflow_start_run() 
  
  #######################################################test###########################
  mlflow_log_param("alpha", alpha)
  mlflow_log_param("lambda", lambda)
  mlflow_log_param("run_name", "glmnet-mgaussian")
  mlflow_log_param("features_p", features)
  model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "mgaussian", standardize = FALSE)
  predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model)
  
  testingPreds <- predictor(test_x)
  validationPreds <- predictor(validation_x)
  trainPreds    <- predictor(train_x)
  
  #Evaluating performance on testing data
  rmse <- sqrt(mean((testingPreds - test_y) ^ 2))
  mae <- mean(abs(testingPreds - test_y))
  r2 <- as.numeric(cor(testingPreds, test_y) ^ 2)
  
  mlflow_log_metric("rmse-testing", rmse)
  mlflow_log_metric("r2-testing", r2)
  mlflow_log_metric("mae-testing", mae)
  
  #Evaluating performance on training data
  rmse <- sqrt(mean((trainPreds - train_y) ^ 2))
  mae <- mean(abs(trainPreds - train_y))
  r2 <- as.numeric(cor(trainPreds, train_y) ^ 2)
  
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
  
  message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
  message("  RMSE: ", rmse)
  message("  MAE: ", mae)
  message("  R2: ", r2)
  
  mlflow_log_model(predictor, "model")


mlflow_end_run()

mlflow_start_run() 
  mlflow_log_param("run_name","lm")
  mlflow_log_param("features_p", features)

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


mlflow_end_run()

mlflow_start_run() 
  #RandomForest
  mlflow_log_param("run_name", "randomforest")
  mlflow_log_param("features_p", features)
  predictorrf <- randomForest(yHat ~ ., data = train, mtry = 10,ntree=500,maxnodes=4,
                              importance = TRUE, na.action = na.omit, proximity=TRUE)
  
  #print(predictorrf)
  trainPreds      <- predict(predictorrf, train)
  validationPreds <- predict(predictorrf, validation)
  testingPreds    <- predict(predictorrf, testing)
  #Evaluating performance on training data
  rmse <- sqrt(mean((trainPreds - train_y) ^ 2))
  mae <- mean(abs(trainPreds - train_y))
  r2 <- as.numeric(cor(trainPreds, train_y) ^ 2)
  
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
#mlflow_ui()

# Make predictions on the prospect file
#prospectsPreds  <- predict(fitLM, prospects)
#write.csv(prospectsPreds, 'prospectPredictionFile.csv', row.names = F)


# End