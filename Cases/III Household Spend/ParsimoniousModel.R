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
#setting the seed as 1234 for reproducability
set.seed(1234) 
#creating a split of Training data into 2 sections, A and B, in order to train and validate the results
idx <- sample(1:nrow(trainingTables), 12000)
trainingTablesSectionA <- trainingTables[idx,]
trainingTablesSectionB <- trainingTables[-idx,]

glimpse(trainingTables)
#wow, this data has so many variables- 80 of them
summary(trainingTables)
str(trainingTables)
plot_missing(trainingTables)
#although plot missing doesn't show a lot of NAs, I notice that there are a lot of blanks in data
trainingNA <- trainingTables                                    # Duplicate data frame
trainingNA[trainingNA == ""] <- NA 
trainingNA[trainingNA == "Unknown"] <- NA 
trainingNA[trainingNA == "Cannot Determine"] <- NA 

# Replace blank by NA
plot_missing(trainingNA)



#looking at count of mising values and using this to eliminate variables which have most values missing
#for eg - DonatestoLiberalCauses - 14921 out of 15000 are empty

#PLot correlations
################


#####

## Explore -- do more exploration 
names(trainingTablesSectionA)
head(trainingTablesSectionA)
barplot(table(trainingTablesSectionA$NetWorth), las = 2)

## Modify 
# Choose which variables are ethical to use, and others which may not be useful; here I just chose 5 variables
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

trainingTablesSectionA$bookbuyerlevel<-as.numeric(trainingTablesSectionA$bookbuyerlevel)

myname<-"manoj"

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
#Iteration # 2, adding few more variables to see if it improves scores, earlier scores were from 92 to 112 (RMSE)


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


mtry=10
ntrees = 500

#Running RandomForest using Ranger and checking the results
rangerrf <- ranger(yHat ~.,data = train, num.trees = ntrees, mtry = mtry)
#print(predictorrf)
trainPreds      <- predict(rangerrf, train)
validationPreds <- predict(rangerrf, validation)
testingPreds    <- predict(rangerrf, testing)

rangerrf
#checking the results only Validation data
rmse <- MLmetrics::RMSE(validationPreds$predictions,validation_y)
mae <- MLmetrics::MAE(validationPreds$predictions,validation_y)
r2 <- MLmetrics::R2_Score(validationPreds$predictions,validation_y)
message("Random Forest - Ranger - Validation (mtry=", mtry, ", ntree=", ntrees,    "):")
message("  RMSE: ", rmse)
message("  MAE: ", mae)
message("  R2: ", r2)

#checking the results for Testing data
rmse <- MLmetrics::RMSE(testingPreds$predictions,test_y)
mae <- MLmetrics::MAE(testingPreds$predictions,test_y)
r2 <- MLmetrics::R2_Score(testingPreds$predictions,test_y)
message("Random Forest - Ranger - Testing (mtry=", mtry, ", ntree=", ntrees,    "):")
message("  RMSE: ", rmse)
message("  MAE: ", mae)
message("  R2: ", r2)


mlf_experiment_id = mlflow_set_experiment(
  experiment_name = "Ranger-loop3"
)


#setting default values for the MLFlow Parameters
features_p <- mlflow_param("features", features, "string")
mtry <- mlflow_param("mtry", 3, "numeric")
ntree <- mlflow_param("ntree", 3, "numeric")
maxnodes<- mlflow_param("maxnodes", 4, "numeric")





#randomForest(yHat ~ ., data = train, mtry = mtry,ntree=ntree,maxnodes=maxnodes,
#  importance = TRUE, na.action = na.omit, proximity=TRUE)

#RandomForest
for (mtry in 3:4) {
  for (ntree in c(seq(from = 50, to = 600, by  = 200))){
    for (maxnodes in 4:5){
      mlflow_start_run()
      mlflow_log_param("run_name", "randomforest")
      mlflow_log_param("features_p", features)
      mlflow_log_param("mtry", mtry)
      mlflow_log_param("ntree", ntree)
      mlflow_log_param("maxnodes", maxnodes)
      
      predictorrf <- ranger(yHat ~.,data = train, num.trees = ntree, mtry = mtry, maxnodes = maxnodes)
      MLmetrics::RMSE(testingPreds$predictions,testing$yHat)
      
      #print(predictorrf)
      trainPreds      <- predict(predictorrf, train)
      validationPreds <- predict(predictorrf, validation)
      testingPreds    <- predict(predictorrf, testing)
      
      #Evaluating performance on training  data
      rmse <- MLmetrics::RMSE(trainPreds$predictions,train_y)
      mae <- MLmetrics::MAE(trainPreds$predictions,train_y)
      r2 <- MLmetrics::R2_Score(trainPreds$predictions,train_y)
      
      mlflow_log_metric("rmse-Train", rmse)
      mlflow_log_metric("r2-Train", r2)
      mlflow_log_metric("mae-Train", mae)
      
      #Evaluating performance on validation data
      rmse <- MLmetrics::RMSE(validationPreds$predictions,validation_y)
      mae <- MLmetrics::MAE(validationPreds$predictions,validation_y)
      r2 <- MLmetrics::R2_Score(validationPreds$predictions,validation_y)
      
      mlflow_log_metric("rmse-Validation", rmse)
      mlflow_log_metric("r2-Validation", r2)
      mlflow_log_metric("mae-Validation", mae)
      
      #Evaluating performance on testing data
      rmse <- MLmetrics::RMSE(testingPreds$predictions,test_y)
      mae <- MLmetrics::MAE(testingPreds$predictions,test_y)
      r2 <- MLmetrics::R2_Score(testingPreds$predictions,test_y)
      
      mlflow_log_metric("rmse-testing", rmse)
      mlflow_log_metric("r2-testing", r2)
      mlflow_log_metric("mae-testing", mae)
      
      message("Random Forest model (mtry=", mtry, ", ntree=", ntree,  "maxnodes = ", maxnodes , "):")
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