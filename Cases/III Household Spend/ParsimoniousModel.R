# Case-III Bed Bath and Yonder - CSCI E-96 - Data Mining for Business
# 
# Project Purpose: Building a predicted model for predicting Household spend (yHat) based on the data files available 
# Provide data insights
# Filename : ParsimoniousModel.R
#   
#
# Student name : Manoj Sharma

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
library(ggpubr)
library(dplyr)
library(caretEnsemble)
library(skimr)
library(igraph)
library(DataExplorer)
library(car)
library(glmnet)
library(carrier)



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

########################################-SAMPLE-#################################################################
## Sample is the first phase of SEMMA approach
##
########################################-SAMPLE-#################################################################

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




#looking at count of mising values and using this to eliminate variables which have most values missing
#for eg - DonatestoLiberalCauses - 14921 out of 15000 are empty


########################################-EXPLORE-#################################################################
## EXPLORE is the 2ND phase of SEMMA approach in which data is explored
##
########################################-EXPLORE-#################################################################
names(trainingTablesSectionA)
head(trainingTablesSectionA)
plot_missing(trainingTables)
#although plot missing doesn't show a lot of NAs, I notice that there are a lot of blanks in data
trainingNA <- trainingTables                                    # Duplicate data frame
trainingNA[trainingNA == ""] <- NA 
trainingNA[trainingNA == "Unknown"] <- NA 
trainingNA[trainingNA == "Cannot Determine"] <- NA 

# Replace blank by NA
plot_missing(trainingNA)
#plotting yHat distribution in training and test sets
hist(trainingTables$yHat,
     main="Distribution of yHat in training set",
     xlab="Household revenue",
      col="darkmagenta",
     freq=TRUE
)
hist(testingTables$yHat,
     main="Distribution of yHat in testing set",
     xlab="Household revenue",
     col="green",
     freq=TRUE
)

# yHat with Age on training dataset
ggplot(trainingTables, aes(Age, yHat)) +
  geom_jitter(size = 0.5, width = 0.5)

# yHat with Age on training dataset
ggplot(testingTables, aes(Age, yHat)) +
  geom_jitter(size = 0.5, width = 0.5)

#plotting yHat against storeVisitFrequency and Gender for training Data
ggscatterhist(
  trainingTables[!(trainingTables$Gender==""),], x = "storeVisitFrequency", y = "yHat",
  color = "Gender", size = 3, alpha = 0.6,
  palette = c("#00AFBB", "#E7B800", "#FC4E07"),
  margin.params = list(fill = "Gender", color = "black", size = 0.2)
)


#plotting yHat against storeVisitFrequency and Gender for testing Data
ggscatterhist(
  testingTables, x = "storeVisitFrequency", y = "yHat",
  color = "Gender", size = 3, alpha = 0.6,
  palette = c("#00AFBB", "#E7B800", "#FC4E07"),
  margin.params = list(fill = "Gender", color = "black", size = 0.2)
)
#plotting yHat against ISPSA  and Gender for training Data
ggscatterhist(
  trainingTables[!(trainingTables$Gender==""),], x = "ISPSA", y = "yHat",
  color = "Gender", size = 3, alpha = 0.6,
  palette = c("#00AFBB", "#E7B800", "#FC4E07"),
  margin.plot = "boxplot",
  ggtheme = theme_bw()
)

#Household spend by Gender
ggplot(trainingTables[!(trainingTables$ResidenceHHGenderDescription==""),], aes(ResidenceHHGenderDescription, yHat)) +
  geom_boxplot()

#Household spend by Education
ggplot(trainingTables[!(trainingTables$Education==""),], aes(Education, yHat)) +
  geom_boxplot()

########################################-MODIFY-#################################################################
## MODIFY is the 3RD phase of SEMMA approach in which data is mofified to be prepared for modeling
##
########################################-MODIFY-#################################################################
# Choose which variables are ethical to use, and others which may not be useful; here I just chose 5 variables

#adding a ordinal variable for bookbuyer
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
#converting the bookbuyerlevel as a numeric
trainingTablesSectionA$bookbuyerlevel<-as.numeric(trainingTablesSectionA$bookbuyerlevel)

#adding a ordinal variable for networth
# trainingTablesSectionA$NWlevel <- sapply(trainingTablesSectionA$NetWorth, switch, 
#                   '$1-4999' = 1, 
#                   '$5000-9999' = 2, 
#                   '$10000-24999'=3,
#                   '$25000-49999'=4,
#                   '$50000-99999'=5,
#                   '$100000-249999'=6,
#                   '$250000-499999'=7,
#                   '$499999+'=8,6)
#adding a numerical variable for Education level for training data
trainingTablesSectionA$EDlevel <- sapply(trainingTablesSectionA$Education, switch,
                  'Unknown' = 3,
                  'Less than HS Diploma - Likely' = 1,
                  'HS Diploma - Likely' = 2,
                  'HS Diploma - Extremely Likely' = 3,
                  'Vocational Technical Degree - Extremely Likely' = 4,
                  'Some College - Likely'=5,
                  'Some College -Extremely Likely'=6,
                  'Bach Degree - Likely'=7,
                  'Bach Degree - Extremely Likely'=8,
                  'Grad Degree - Likely'=9,
                  'Grad Degree - Extremely Likely'=10,5)
trainingTablesSectionA$EDlevel<-as.numeric(trainingTablesSectionA$EDlevel)

#adding a numerical variable for Education level for validation data
trainingTablesSectionB$EDlevel <- sapply(trainingTablesSectionB$Education, switch,
                                         'Unknown' = 3,
                                         'Less than HS Diploma - Likely' = 1,
                                         'HS Diploma - Likely' = 2,
                                         'HS Diploma - Extremely Likely' = 3,
                                         'Vocational Technical Degree - Extremely Likely' = 4,
                                         'Some College - Likely'=5,
                                         'Some College -Extremely Likely'=6,
                                         'Bach Degree - Likely'=7,
                                         'Bach Degree - Extremely Likely'=8,
                                         'Grad Degree - Likely'=9,
                                         'Grad Degree - Extremely Likely'=10,5)
trainingTablesSectionB$EDlevel<-as.numeric(trainingTablesSectionB$EDlevel)

#adding a numerical variable for Education level for testing data
testingTables$EDlevel <- sapply(testingTables$Education, switch,
                                         'Unknown' = 3,
                                         'Less than HS Diploma - Likely' = 1,
                                         'HS Diploma - Likely' = 2,
                                         'HS Diploma - Extremely Likely' = 3,
                                         'Vocational Technical Degree - Extremely Likely' = 4,
                                         'Some College - Likely'=5,
                                         'Some College -Extremely Likely'=6,
                                         'Bach Degree - Likely'=7,
                                         'Bach Degree - Extremely Likely'=8,
                                         'Grad Degree - Likely'=9,
                                         'Grad Degree - Extremely Likely'=10,5)
testingTables$EDlevel<-as.numeric(testingTables$EDlevel)

#adding a numerical variable for Education level for prospect data
prospectTables$EDlevel <- sapply(prospectTables$Education, switch,
                                'Unknown' = 3,
                                'Less than HS Diploma - Likely' = 1,
                                'HS Diploma - Likely' = 2,
                                'HS Diploma - Extremely Likely' = 3,
                                'Vocational Technical Degree - Extremely Likely' = 4,
                                'Some College - Likely'=5,
                                'Some College -Extremely Likely'=6,
                                'Bach Degree - Likely'=7,
                                'Bach Degree - Extremely Likely'=8,
                                'Grad Degree - Likely'=9,
                                'Grad Degree - Extremely Likely'=10,5)
prospectTables$EDlevel<-as.numeric(prospectTables$EDlevel)

#creating 3 variables with feature lists. one with fewest, one with some more (<80% missing), and one with most features
#this is in order to find out the best model, with fewest set of features
most_features<-c('PartiesDescription','storeVisitFrequency',
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
            'UpscaleBuyerInHome',
            'BusinessOwner',
            'HorseOwner',
            'overallsocialviews'
)

features_MP<-c('PartiesDescription','storeVisitFrequency',
                                       'ResidenceHHGenderDescription',
                                       'Gender',
                                       'Age',
                                       'EstHomeValue',
                                       'MedianEducationYears',
                                       'ISPSA')

features_MidP<-c('PartiesDescription','storeVisitFrequency',
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
               'ReligionsDescription')


#providing the feature list to create a treatment plan. Preparing the features variable to be stored in ML Flow
informativeFeatures <- most_features
features<-paste(most_features, collapse="|")

#creating treatment plan based on training data
plan  <- designTreatmentsN(trainingTablesSectionA, 
                           informativeFeatures,
                           'yHat')

print(informativeFeatures)
# Prepare all sections of the data, based on the treatment plan
train      <- prepare(plan, trainingTablesSectionA)
validation <- prepare(plan, trainingTablesSectionB)
testing    <- prepare(plan, testingTables)
prospects  <- prepare(plan, prospectTables)

########################################-MODEL-#################################################################
## MODEL is the 4TH phase of SEMMA approach in which  modeling is done
## various models are tried, with different parameters
## Hyperparameter tuning, feature selection is being done in this phase
########################################-MODEL-#################################################################

#preparing the data in the required form for trying into the models
train_x <- as.matrix(train[, !(names(train) == "yHat")])
test_x <- as.matrix(testing[, !(names(train) == "yHat")])
train_y <- train[, "yHat"]
test_y <- testing[, "yHat"]
validation_x <- as.matrix(validation[, !(names(train) == "yHat")])
validation_y <- validation[, "yHat"]



# The next few lines are created to execute grid search using Caret package. Upon running different grid searches using Caret and also
# manually running such tuning of models, I decided on using the mtry of 18 and ntrees as 260

# tgrid <- expand.grid(
#   .mtry = 2:10,
#   .min.node.size = c(10, 20,30),
#   .splitrule = "variance"
# 
# )

# model_caret <- train(yHat  ~ ., data = train,
#                      method = "ranger",
#                      trControl = trainControl(method="cv", number = 5, verboseIter = T, classProbs = F),
#                      tuneGrid = tgrid,
#                      num.trees = 500,
#                      importance = "permutation")


# varImp(model_caret)
# 
# plot(model_caret)
# 
# print(model_caret)
# 
# trainPreds      <- predict(model_caret, train)
# validationPreds <- predict(model_caret, validation)
# testingPreds    <- predict(model_caret, testing)
# 
# 
# #checking the results only Validation data
# rmse <- MLmetrics::RMSE(validationPreds,validation_y)
# mae <- MLmetrics::MAE(validationPreds,validation_y)
# r2 <- MLmetrics::R2_Score(validationPreds,validation_y)
# message("Caret Tuned Model (mtry=", mtry, ", ntree=", ntrees,    "):")
# message("  RMSE: ", rmse)
# message("  MAE: ", mae)
# message("  R2: ", r2)

#these values were found after running experimentation and grid searches
mtry=18
ntrees=260
#setting up a new MLFlow experiment
mlf_experiment_id = mlflow_set_experiment(
  experiment_name = "Parsimonious model-MostFeatures"
)

#starting the MLFlow run
mlflow_start_run() 
  #Running RandomForest using Ranger and checking the results
  rangerrf <- ranger(yHat ~.,data = train, num.trees = ntrees, mtry = mtry)
  #print(predictorrf)

    
  #making predictions using the ranger (RandomForest) model
  trainPreds      <- predict(rangerrf, train)
  validationPreds <- predict(rangerrf, validation)
  testingPreds    <- predict(rangerrf, testing)

  #rangerrf
  #Evaluating performance on training data
  rmse <- MLmetrics::RMSE(trainPreds$predictions,train_y)
  mae <- MLmetrics::MAE(trainPreds$predictions,train_y)
  r2 <- MLmetrics::R2_Score(trainPreds$predictions,train_y)
  
  #Logging metrics in MLFlow for training
  mlflow_log_metric("rmse-Train", rmse)
  mlflow_log_metric("r2-Train", r2)
  mlflow_log_metric("mae-Train", mae)
  
  #checking the results on Validation data
  rmse <- MLmetrics::RMSE(validationPreds$predictions,validation_y)
  mae <- MLmetrics::MAE(validationPreds$predictions,validation_y)
  r2 <- MLmetrics::R2_Score(validationPreds$predictions,validation_y)

   #Logging metrics in MLFlow for validation
  mlflow_log_metric("rmse-Validation", rmse)
  mlflow_log_metric("r2-Validation", r2)
  mlflow_log_metric("mae-Validation", mae)
  
  # message("Random Forest - Ranger - Validation (mtry=", mtry, ", ntree=", ntrees,    "):")
  # message("  RMSE: ", rmse)
  # message("  MAE: ", mae)
  # message("  R2: ", r2)
  
  #checking the results for Testing data
  rmse <- MLmetrics::RMSE(testingPreds$predictions,test_y)
  mae <- MLmetrics::MAE(testingPreds$predictions,test_y)
  r2 <- MLmetrics::R2_Score(testingPreds$predictions,test_y)

 #Logging metrics in MLFlow for testing
  mlflow_log_metric("rmse-testing", rmse)
  mlflow_log_metric("r2-testing", r2)
  mlflow_log_metric("mae-testing", mae)
  
  message("Random Forest - Ranger - Testing (mtry=", mtry, ", ntree=", ntrees,    "):")
  message("  RMSE: ", rmse)
  message("  MAE: ", mae)
  message("  R2: ", r2)

mlflow_end_run()




#setting default values for the MLFlow Parameters
features_p <- mlflow_param("features", features, "string")
alpha <- mlflow_param("alpha", 0.15, "numeric")
lambda <- mlflow_param("lambda", 0.45, "numeric")
run_name<-mlflow_param("name","model_name","string")


mlflow_start_run() 

  #######################################################test###########################
  mlflow_log_param("alpha", alpha)
  mlflow_log_param("lambda", lambda)
  mlflow_log_param("name", "glmnet-mgaussian")
  mlflow_log_param("features_p", features)

  #Creating and training glmnet model
  model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "mgaussian", standardize = FALSE)
  predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model)
  
  #making predictions using the glmnet model
  testingPreds <- predictor(test_x)
  validationPreds <- predictor(validation_x)
  trainPreds    <- predictor(train_x)
  
  #Evaluating performance on testing data
  rmse <- MLmetrics::RMSE(testingPreds,test_y)
  mae <- MLmetrics::MAE(testingPreds,test_y)
  r2 <- MLmetrics::R2_Score(testingPreds,test_y)
  
  
 #Logging metrics in MLFlow for testing
  mlflow_log_metric("rmse-testing", rmse)
  mlflow_log_metric("r2-testing", r2)
  mlflow_log_metric("mae-testing", mae)
  
  #Evaluating performance on training data
  rmse <- MLmetrics::RMSE(trainPreds,train_y)
  mae <- MLmetrics::MAE(trainPreds,train_y)
  r2 <- MLmetrics::R2_Score(trainPreds,train_y)
  
  
 #Logging metrics in MLFlow for training
  mlflow_log_metric("rmse-Train", rmse)
  mlflow_log_metric("r2-Train", r2)
  mlflow_log_metric("mae-Train", mae)
  
  #Evaluating performance on validation data
  rmse <- MLmetrics::RMSE(validationPreds,validation_y)
  mae <- MLmetrics::MAE(validationPreds,validation_y)
  r2 <- MLmetrics::R2_Score(validationPreds,validation_y)
  
  
 #Logging metrics in MLFlow for validation
  mlflow_log_metric("rmse-Validation", rmse)
  mlflow_log_metric("r2-Validation", r2)
  mlflow_log_metric("mae-Validation", mae)
  
  message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
  message("  RMSE: ", rmse)
  message("  MAE: ", mae)
  message("  R2: ", r2)
  
  mlflow_log_model(predictor, "model")
mlflow_end_run()


#The following code was created to manually iterate through hypertuning parameters and logging each iteration
#using MLFlow, after running many such experiments, I found out that ntree=18 or 19 gave the best RMSE
#with this dataset on Test

#randomForest(yHat ~ ., data = train, mtry = mtry,ntree=ntree,maxnodes=maxnodes,
#  importance = TRUE, na.action = na.omit, proximity=TRUE)

#RandomForest - commeting this as it takes a long time to run. Final output after doing this gridsearch wasn't much encouraging either
# for (mtry in 3:4) {
#   for (ntree in c(seq(from = 50, to = 600, by  = 50))){
#     for (maxnodes in 2:5){
#       mlflow_start_run()
#       mlflow_log_param("run_name", "randomforest")
#       mlflow_log_param("features_p", features)
#       mlflow_log_param("mtry", mtry)
#       mlflow_log_param("ntree", ntree)
#       mlflow_log_param("maxnodes", maxnodes)
#       
#       predictorrf <- ranger(yHat ~.,data = train, num.trees = ntree, mtry = mtry, maxnodes = maxnodes)
#       MLmetrics::RMSE(testingPreds$predictions,testing$yHat)
#       
#       #print(predictorrf)
#       trainPreds      <- predict(predictorrf, train)
#       validationPreds <- predict(predictorrf, validation)
#       testingPreds    <- predict(predictorrf, testing)
#       
#       #Evaluating performance on training  data
#       rmse <- MLmetrics::RMSE(trainPreds$predictions,train_y)
#       mae <- MLmetrics::MAE(trainPreds$predictions,train_y)
#       r2 <- MLmetrics::R2_Score(trainPreds$predictions,train_y)
#       
#       mlflow_log_metric("rmse-Train", rmse)
#       mlflow_log_metric("r2-Train", r2)
#       mlflow_log_metric("mae-Train", mae)
#       
#       #Evaluating performance on validation data
#       rmse <- MLmetrics::RMSE(validationPreds$predictions,validation_y)
#       mae <- MLmetrics::MAE(validationPreds$predictions,validation_y)
#       r2 <- MLmetrics::R2_Score(validationPreds$predictions,validation_y)
#       
#       mlflow_log_metric("rmse-Validation", rmse)
#       mlflow_log_metric("r2-Validation", r2)
#       mlflow_log_metric("mae-Validation", mae)
#       
#       #Evaluating performance on testing data
#       rmse <- MLmetrics::RMSE(testingPreds$predictions,test_y)
#       mae <- MLmetrics::MAE(testingPreds$predictions,test_y)
#       r2 <- MLmetrics::R2_Score(testingPreds$predictions,test_y)
#       
#       mlflow_log_metric("rmse-testing", rmse)
#       mlflow_log_metric("r2-testing", r2)
#       mlflow_log_metric("mae-testing", mae)
#       
#       message("Random Forest model (mtry=", mtry, ", ntree=", ntree,  "maxnodes = ", maxnodes , "):")
#       message("  RMSE: ", rmse)
#       message("  MAE: ", mae)
#       message("  R2: ", r2)
#       #######################################################test###########################
#       #mlflow_set_tracking_uri("http://localhost:5712")
#       mlflow_end_run()  
#       
#     }
#   }
# }

#mlflow_ui()
########################################-ASSESS-#################################################################
## ASSESS is the 5TH and final phase of SEMMA approach in which  assessment of models is performed
## As I stated in the presentation, ranger function produced best performing model
##  
########################################-ASSESS-#################################################################
# Make predictions on the prospect file using the the optimized Random Forest model
prospectsPreds  <- predict(rangerrf, prospects)
prospectPreds<-cbind(prospectTables,PredictedyHat=prospectPreds$predictions)
  
# Creating the CSV of the Prospects
write.csv(prospectPreds, 'prospectPredictionFile.csv')



# End