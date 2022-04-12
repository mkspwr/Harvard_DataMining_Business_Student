# Case-II OK Cupid - CSCI E-96 - Data Mining for Business
# For National City Bank use only
# Purpose: Create a customer propensity model for a new product, specifically a line of credit
# against a household's used car.
# Bank has historical data from 4000 previous calls and mailings for the line of credit offer. Using this data
# I prepared prediction models using various approaches to shortlist customer IDs from a list of 1000 
# prospective customers
# Student name : Manoj Sharma

#Loading R libraries for the analysis

library(Matrix)
library(plyr)
library(dplyr)
library(data.table)
library(tidyverse)
library(DataExplorer)
library(vtreat)
library(MLmetrics)
library(car)
library(chron)
library(caret)
library(pROC)
library(e1071)
library(randomForest)
library(rpart)
library(rpart.plot)
library(scales)
library(ggplot2)
library(class)

#Importing datafiles 
setwd("C:/Users/mksharma/Harvard/Harvard_DataMining_Business_Student") #setting working  directory
setwd("./Cases/II National City Bank/training") #setting working folder
currentData   <- read.csv('CurrentCustomerMktgResults.csv')
vehicleData <- read.csv('householdVehicleData.csv') 
axiomData <- read.csv('householdAxiomData.csv')
creditData <- read.csv('householdCreditData.csv')
carInsurance_test<-read.csv('carInsurance_test.csv')
setwd("C:/Users/mksharma/Harvard/Harvard_DataMining_Business_Student")
setwd("./Cases/II National City Bank")
inputProspect<-read.csv('prospectiveCustomers.csv') #importing prospect data, for eventual prediction 


#joining training data with Vehicle, Axiom and Credit using HHuniqueID field using left_join
joinData <- left_join(currentData,vehicleData,by = c('HHuniqueID'))
joinData <- left_join(joinData, axiomData, by = c('HHuniqueID'))
joinData <- left_join(joinData, creditData, by = c('HHuniqueID'))

#joining prospects data with Vehicle, Axiom and Credit using HHuniqueID field using left_join
prospectdata <- left_join(inputProspect,vehicleData,by = c('HHuniqueID'))
prospectdata <- left_join(prospectdata, axiomData, by = c('HHuniqueID'))
prospectdata <- left_join(prospectdata, creditData, by = c('HHuniqueID'))
#since the prospects data doesn't have call_start and call_end fields, 
#I am bringing in from the external data, from the file called carInsurance_test.csv
prospectdata<-left_join(prospectdata,carInsurance_test, by = c('dataID'='Id'))
#processing the prospectdata and processing it in the same way as training data, to be consistent 


###############################SAMPLE##################################################################
#Using SEMMA based approach..as taught in the class
#SEMMA - Sample - Explore - Modify- Model - Analyze
#SAMPLING DATA
###############################SAMPLE##################################################################

head(joinData)
tail(joinData)
names(joinData)
names(prospectdata)
str(joinData)

plot_missing(joinData)

###############################EXPLORE##################################################################
#Using SEMMA based approach..as taught in the class
#SEMMA - Sample - Explore - Modify- Model - Analyze
#EXPLORING DATA
###############################EXPLORE##################################################################

#Data exploration leads to a plan for treating
summary(joinData)
summary(prospectdata)
#Exploration tells me the following
# 1. Some columns got duplicated, which will need to be removed
# 2. Will need to treat for missing values which show as NA in Past_Outcome, Communication, CarYr, Education and Job columns
# 3. Call start and Call end can be combined to calculate call duration, (feature engineering)
# 4. the data appears to have day and month, will need to impute year, in order to feature-engineer a date column, 
# 5. call start is a continuous variable, will need to try identifying if call times explain Acceptance of an offer

###############################MODIFY##################################################################
#Using SEMMA based approach..as taught in the class
#SEMMA - Sample - Explore - Modify- Model - Analyze
#MODIFYING DATA
###############################MODIFY##################################################################


#removing duplicated columns
clean_prospect<-subset(prospectdata, select = -c(Age.y,Job.y,Marital.y,Education.y,HHInsurance.y,CarLoan.y,
                                                 Communication.y,LastContactDay.y,LastContactMonth.y, NoOfContacts.y, DaysPassed.y,
                                                 PrevAttempts.y, Balance,CarInsurance,DefaultOnRecord,Outcome))

names(clean_prospect)
str(clean_prospect)
names(joinData)
str(joinData)
plot_missing(joinData)
plot_missing(clean_prospect)
trainData<- joinData



clean_d <- joinData


#Part of the data-cleaning and imputations have been taken from the code example at the following location:
# https://www.kaggle.com/kondla/simple-random-forest-on-insurance-call-forecast/code

#imputing year as 2020 not sure if the data is from leap year or not, to be safe..and, car models have a mode of 2019, hence choosing 2020
#Although the Kaggle code example imputed 2015 for the Year, it didn't make sense, because there were car Yr of 2019. Hence, I decided
#to impute the year as 2020
clean_d$DateCall <- as.Date(paste(clean_d$LastContactDay, clean_d$LastContactMonth, "2020", sep = '/'), "%d/%b/%Y")  
clean_d$Weekday <- factor(weekdays(clean_d$DateCall))

clean_prospect$DateCall <- as.Date(paste(clean_prospect$LastContactDay, clean_prospect$LastContactMonth, "2020", sep = '/'), "%d/%b/%Y")  
clean_prospect$Weekday <- factor(weekdays(clean_prospect$DateCall))

plot(table(clean_d$CallStart)) # 
plot(table(call_hr <- gsub("(:\\d{2})", "", clean_d$CallStart)))

# We could take the times as they are given. However, that would be too much noise, in my opinion. Therefore, I opt for three time slots, i.e. morning 9 - 11:59:59, midday 12 - 14:59:59, afternoon the rest
clean_d$CallDayTime <- as.numeric(gsub("(:\\d{2})", "", clean_d$CallStart))
clean_prospect$CallDayTime <- as.numeric(gsub("(:\\d{2})", "", clean_prospect$CallStart))




clean_d$call_dur_min <- 60 * 24 * as.numeric(times(clean_d$CallEnd)-times(clean_d$CallStart))
clean_prospect$call_dur_min <- 60 * 24 * as.numeric(times(clean_prospect$CallEnd)-times(clean_prospect$CallStart))




#na_count <-sapply(clean_d, function(y) sum(length(which(is.na(y)))))
#na_count <- data.frame(na_count)


summary(clean_d)
names(clean_d)
str(clean_d)
names(clean_prospect)
str(clean_prospect)

clean_prospect <-clean_prospect %>%
  rename(
    Communication = Communication.x,
    LastContactDay = LastContactDay.x,
    LastContactMonth =  LastContactMonth.x,
    NoOfContacts = NoOfContacts.x,
    DaysPassed = DaysPassed.x,
    PrevAttempts = PrevAttempts.x,
    Age = Age.x,
    Job = Job.x,
    Marital = Marital.x,
    Education = Education.x,
    HHInsurance = HHInsurance.x,
    CarLoan = CarLoan.x,
    DefaultOnRecord = Default
  )
          
 

sub_clean_d <- subset(clean_d, select = -c(dataID, LastContactDay, LastContactMonth,  CallStart, CallEnd, DateCall))

#removing the education variable which has been imputed earlier
sub_cleanProspect<-subset(clean_prospect, select = -c(dataID,LastContactDay,LastContactMonth,CallStart, CallEnd, DateCall ))
#Renaming the variables to be consistent



names(sub_clean_d)
names(sub_cleanProspect)
model_prospect = select(sub_cleanProspect, "HHuniqueID","Communication","NoOfContacts","DaysPassed","PrevAttempts",
                           "carMake","carModel",
                           "carYr","headOfhouseholdGender",   
                           "annualDonations","EstRace",
                           "PetsPurchases","DigitalHabits_5_AlwaysOn",
                           "AffluencePurchases","Age",
                           "Job","Marital",
                           "Education","DefaultOnRecord",
                           "RecentBalance","HHInsurance",
                           "CarLoan","Weekday",
                           "CallDayTime","past_Outcome","call_dur_min","Y_AcceptedOffer" )

model_d = select(sub_clean_d,"HHuniqueID","Communication","NoOfContacts","DaysPassed","PrevAttempts",
                 "carMake","carModel",
                 "carYr","headOfhouseholdGender",   
                 "annualDonations","EstRace",
                 "PetsPurchases","DigitalHabits_5_AlwaysOn",
                 "AffluencePurchases","Age",
                 "Job","Marital",
                 "Education","DefaultOnRecord",
                 "RecentBalance","HHInsurance",
                 "CarLoan","Weekday",
                 "CallDayTime","past_Outcome","call_dur_min","Y_AcceptedOffer")


table(model_d$carYr)
#function to find Mode, to impute Mode for NAs on car Year
Mode <- function(x) {
   ux <- unique(x)
   tab <- tabulate(match(x, ux))
   ux[tab == max(tab)]
}
mode_carYr<-Mode(model_d$carYr) #finding mode, turns out that mode in this case is 2019

train_control = trainControl(method="cv", number=10)

plot_missing(model_d)
model_d<-model_d %>%
  mutate(Job=ifelse(is.na(Job),"not provided",Job)) %>%
  mutate(Communication=ifelse(is.na(Communication),"not provided",Communication)) %>%
  mutate(carYr=ifelse(is.na(carYr),mode_carYr,carYr)) %>%
  mutate(Y_AcceptedOffer=ifelse(Y_AcceptedOffer=="Accepted",1,0))


model_prospect<-model_prospect %>%
  mutate(Job=ifelse(is.na(Job),"not provided",Job)) %>%
  mutate(Communication=ifelse(is.na(Communication),"not provided",Communication)) %>%
  mutate(carYr=ifelse(is.na(carYr),mode_carYr,carYr)) %>%
  mutate(Y_AcceptedOffer=ifelse(Y_AcceptedOffer=="Accepted",1,0))


plot_missing(model_d) 
str(model_d)

plot_missing(model_prospect)
str(model_prospect)

table(model_d$Marital)
table(model_prospect$Marital)


names(model_d)
str(model_d)

#visualizing training data
ggplot(joinData,aes(x=headOfhouseholdGender,fill=Y_AcceptedOffer)) +
  geom_histogram(stat="count")

#visualizing prospects data, in order to make the model work, the data needs to be similar
ggplot(joinData,aes(x=headOfhouseholdGender,fill=Y_AcceptedOffer)) +
  geom_histogram(stat="count")


ggplot(joinData,aes(x=Marital, fill=Y_AcceptedOffer)) +
  geom_histogram(stat="count")

ggplot(joinData,aes(x=carMake, fill=Y_AcceptedOffer)) +
  geom_histogram(stat="count")

ggplot(joinData,aes(x=Age, colour=Y_AcceptedOffer)) +
  geom_histogram(stat="count")

ggplot(joinData,aes(x=Age,stat(density), colour=Y_AcceptedOffer)) +
  geom_histogram(binwidth=5)



names(model_d)
str(model_d)
names(model_prospect)
str(model_prospect)
head(model_d)
plot_missing(model_d)
plot_missing(model_prospect)
###############################MODEL##################################################################
#Using SEMMA based approach..as taught in the class
#SEMMA - Sample - Explore - Modify- Model - Analyze
#At this point my data preparation is complete, and I am ready to treat the data, split in train and test, and model
#Creating the model begins now
###############################MODEL###################################################################

# Treatment of data for the model using vtreat
targetVar       <- names(model_d)[27]
informativeVars <- names(model_d)[2:26]

# Design a "C"ategorical variable plan 
plan <- designTreatmentsC(model_d, 
                          informativeVars,
                          targetVar,1)
#applying treatment to the training data
treatedmodel_d <- prepare(plan, model_d)

#applying the same treatment to the prospects data
treatedmodel_prospect <- prepare(plan, model_prospect)


set.seed(1234) #setting the seed for reproducibility
train_index <- createDataPartition(treatedmodel_d$Y_AcceptedOffer, p = 0.85, list = FALSE, times = 1) #using 80% for training and 
training <- treatedmodel_d[train_index, ]
testing <- treatedmodel_d[-train_index, ] # remaining 20% for Testing the accuracy

plot_missing(training)
plot_missing(testing)
table(training$Y_AcceptedOffer)
table(testing$Y_AcceptedOffer)
#######################################################################################################################
#RUNNING Decision Tree Model
########################################################################################################################


#Decision tree
dfit <- train(as.factor(Y_AcceptedOffer) ~., #formula based
              data = training, #data in
              method = "rpart", 
              tuneGrid = data.frame(cp = c(0.0001, 0.001,0.01, 0.05, 0.07)), 
              control = rpart.control(minsplit = 1, minbucket = 4))

overFit<-rpart(as.factor(Y_AcceptedOffer)~.,
               data=training,
               method="class",
               minsplit = 1,
               minbucket = 1,
               cp=-1)

prp(overFit,extra = 1)
#making predictions using Caret
trainCaret <- predict(dfit, training)
testCaret<-predict(dfit,testing)
prospectDT<-predict(dfit,treatedmodel_prospect,type='prob')

#adding the DT prediction to the entire training data
trainingDT<-predict(dfit,treatedmodel_d,type='prob')
trainingDT<-cbind(model_d,DTPrediction=trainingDT)

#adding the decision tree prediction as a column in the prospect datafrae
inputProspect<-cbind(inputProspect,DTPrediction = prospectDT)
# Get the conf Matrix
confusionMatrix(trainCaret, as.factor(training$Y_AcceptedOffer),positive='1')
confusionMatrix(testCaret, as.factor(testing$Y_AcceptedOffer),positive='1')

#exporting the preparted data to run AutoML on AzureML just for experimentation
write.csv(training,file="training.csv")
write.csv(testing,file='testing.csv')

#######################################################################################################################
#RUNNING Random Forest Tree Model
########################################################################################################################


# Training the actual model. We have to pass CarInsurance, i.e. the variable to be classified, as a factor, not as a numeric. Otherwise the model gets confused and thinks we want to create a regression prediction instead of a binary classification
set.seed(1234)
treatedTrain<-training
treatedTest<- testing
plot_missing(treatedTrain)

Sys.time()
#Training RandomForest
downSampleFit <- train(Y_AcceptedOffer ~ .,
                       data = treatedTrain,
                       method = "rf",
                       verbose = FALSE,
                       ntree = 300,
                       tuneGrid = data.frame(mtry = 27)) #num of vars used in each tree
downSampleFit

predClasses <- predict(downSampleFit,  treatedTrain)


#predClasses
cutoff=0.5
predAccept_train <- ifelse(predClasses >= cutoff, 1,0)
results_DSF <- data.frame(actual  = treatedTrain$Y_AcceptedOffer,
                            team    = treatedTrain$Communication_catP,
                            seed    = treatedTrain$NoOfContacts,
                            classes = predAccept_train,
                            probs   = predClasses)


ggplot(results_DSF, aes(x=probs, color=as.factor(actual))) +
  geom_density() + 
  geom_vline(aes(xintercept = cutoff), color = 'green')

# Confusion Matrix
caret::confusionMatrix(factor(predAccept_train), 
                       factor(treatedTrain$Y_AcceptedOffer),positive='1')

prospectRF1<-predict(downSampleFit,treatedmodel_prospect)


cutoff=0.5
inputProspect<-cbind(inputProspect,RF1Prediction = prospectRF1)
write.csv(inputProspect,file="predictedProspect.csv")



#adding the RF prediction to the entire training data
trainingRF1<-predict(downSampleFit,treatedmodel_d)
trainingDT<-cbind(trainingDT,RF1Prediction=trainingRF1)


# Other interesting model artifacts
varImp(downSampleFit)
plot(varImp(downSampleFit), top = 20)

#Arrived at the ntree and mtry parameters by experimenting and running the models in a loop
# Add more trees to the forest with the randomForest package (caret takes a long time bc its more thorough)
moreVoters <- randomForest(as.factor(Y_AcceptedOffer) ~ .,
                           data  = treatedTrain, 
                           ntree = 500,
                           mtry  = 27)

# Confusion Matrix, compare to 3 trees ~63% accuracy
trainClass <- predict(moreVoters, treatedTrain)
confusionMatrix(trainClass, as.factor(treatedTrain$Y_AcceptedOffer),positive="1")


prospectRF500<-predict(moreVoters,treatedmodel_prospect,type='prob')
inputProspect<-cbind(inputProspect,prospectRF500 = prospectRF500)


#adding the RF prediction to the entire training data
trainingRF500<-predict(moreVoters,treatedmodel_d,type='prob')
trainingDT<-cbind(trainingDT,RF500rediction=trainingRF500)


# Look at improved var importance
varImpPlot(moreVoters)

# Out of Bag OOB= avg prediction error on each training sample using trees that weren't built with those records (similar to a validation)
#https://en.wikipedia.org/wiki/Out-of-bag_error

# plot the RF with a legend
# https://stackoverflow.com/questions/20328452/legend-for-random-forest-plot-in-r
layout(matrix(c(1,2),nrow=1),
       width=c(4,1)) 
par(mar=c(5,4,4,0)) #No margin on the right side
plot(moreVoters, log="y")
par(mar=c(5,0,4,2)) #No margin on the left side
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(moreVoters$err.rate),col=1:4,cex=0.8,fill=1:4)


# Let's optimize # of trees 
someVoters <- randomForest(as.factor(Y_AcceptedOffer) ~ .,
                           data = treatedTrain, 
                           ntree=100,
                           mtry = 15)

fewvoters <- randomForest(as.factor(Y_AcceptedOffer) ~ .,
                           data = treatedTrain, 
                           ntree=3,
                           mtry = 9)

# Confusion Matrix
trainClass <- predict(someVoters, treatedTrain)
confusionMatrix(trainClass, as.factor(treatedTrain$Y_AcceptedOffer),positive='1')

prospectRF100<-predict(someVoters,treatedmodel_prospect,type='prob')
inputProspect<-cbind(inputProspect,prospectRF100 = prospectRF100)


#adding the RF100 prediction to the entire training data
trainingRF100<-predict(someVoters,treatedmodel_d,type='prob')
trainingDT<-cbind(trainingDT,RF100rediction=trainingRF100)

### Now let's apply to the validation test set
threeVotes        <- predict(fewvoters, treatedTest)
fiveHundredVoters <- predict(moreVoters,    treatedTest)
oneHundredVoters  <- predict(someVoters,    treatedTest)

# Accuracy Comparison from MLmetrics
Accuracy(treatedTest$Y_AcceptedOffer, threeVotes)
Accuracy(treatedTest$Y_AcceptedOffer, fiveHundredVoters)
Accuracy(treatedTest$Y_AcceptedOffer, oneHundredVoters)

Sys.time() #checking start time and end time to find out how long does it take to run the model
model_rf = train(factor(Y_AcceptedOffer)~., data=treatedTrain, trControl=train_control, method="rf")
Sys.time()
# We make a frame and fill it with the predicted values. This allows us to the test the quality of the model in the next step
prediction_rf = predict(model_rf, subset(treatedTest, select=-c(Y_AcceptedOffer)))

prediction_rf_train = predict(model_rf, subset(treatedTrain, select=-c(Y_AcceptedOffer)))

prospectRFtc<-predict(model_rf,treatedmodel_prospect,type='prob')
inputProspect<-cbind(inputProspect,prospectRFtc = prospectRFtc)

#adding the RFtc prediction to the entire training data
trainingRFtc<-predict(model_rf,treatedmodel_d,type='prob')
trainingDT<-cbind(trainingDT,trainingRFtc=trainingRFtc)

#Compute the accuracy of predictions with a confusion matrix
confusionMatrix(prediction_rf_train,positive="1", as.factor(treatedTrain$Y_AcceptedOffer))

confusionMatrix(prediction_rf,positive="1", as.factor(treatedTest$Y_AcceptedOffer))
#######################################################################################################################
#RUNNING Logistic Regression model
########################################################################################################################



set.seed(1234)
model_logreg <- glm(factor(factor(Y_AcceptedOffer)) ~., family=binomial(link='logit'), data=treatedTrain)

bestFit <- step(model_logreg, direction='backward')
saveRDS(bestFit, 'bestFit.rds')
bestFit <- readRDS('bestFit.rds')
summary(bestFit)

# Compare model size
length(coefficients(fit))
length(coefficients(bestFit))

prediction_logreg = predict(model_logreg, treatedTest, type='response')

prediction_logreg_train = predict(model_logreg, treatedTrain, type='response')


lgcutoff      <- 0.35
predAccept_train <- ifelse(prediction_logreg_train >= lgcutoff, 1,0)
predAccept <- ifelse(prediction_logreg >= lgcutoff, 1,0)

prospectLogReg = predict(model_logreg,treatedmodel_prospect,type='response')

inputProspect<-cbind(inputProspect,prospectLogReg = prospectLogReg )

#adding the RFtc prediction to the entire training data
trainingLG<-predict(model_logreg,treatedmodel_d,type='response')
trainingDT<-cbind(trainingDT,trainingLG=trainingLG)

# Organize w/Actual
results_train <- data.frame(actual  = treatedTrain$Y_AcceptedOffer,
                      team    = treatedTrain$Communication_catP,
                      seed    = treatedTrain$NoOfContacts,
                      classes = predAccept_train,
                      probs   = prediction_logreg_train)

results_test <- data.frame(actual  = treatedTest$Y_AcceptedOffer,
                           team    = treatedTest$Communication_catP,
                           seed    = treatedTest$NoOfContacts,
                           classes = predAccept,
                           probs   = prediction_logreg)

head(results_train)

# Get a confusion matrix
#for Training
(confMat <- ConfusionMatrix(results_train$classes, results_train$actual))
#for Test
(confMat_test <- ConfusionMatrix(results_test$classes, results_test$actual))


# What is the accuracy?
sum(diag(confMat)) / sum(confMat)
sum(diag(confMat_test)) / sum(confMat_test)

# This is the actual KPI Accuracy not to be confused with the forecast package function accuracy() which the book uses :(
Accuracy(results_train$classes, results_train$actual)

Accuracy(results_test$classes, results_test$actual)




# Visually how well did we separate our classes?
ggplot(results_train, aes(x=probs, color=as.factor(actual))) +
  geom_density() + 
  geom_vline(aes(xintercept = cutoff), color = 'green')

ggplot(results_test, aes(x=probs, color=as.factor(actual))) +
  geom_density() + 
  geom_vline(aes(xintercept = cutoff), color = 'green')

# ROC
ROCobj <- roc(results_test$classes, results_test$actual)
plot(ROCobj)

# AUC
AUC(results_test$actual,results_test$classes)

#######################################################################################################################
#RUNNING Logitboost model, to see if this does any better than the previous ones
########################################################################################################################


  
set.seed(1234)
model_logitboost <- train(factor(Y_AcceptedOffer)~., data=treatedTrain, trainControl=train_control, method="LogitBoost", nIter=50)

prediction_logitboost_train = predict(model_logitboost, subset(treatedTrain, select=-c(Y_AcceptedOffer)))

prediction_logitboost = predict(model_logitboost, subset(treatedTest, select=-c(Y_AcceptedOffer)))

confusionMatrix(prediction_logitboost_train,as.factor(treatedTrain$Y_AcceptedOffer),positive="1")

confusionMatrix(prediction_logitboost,as.factor(treatedTest$Y_AcceptedOffer),positive="1")

prospectLGBoost = predict(model_logitboost,treatedmodel_prospect,type='prob')

inputProspect<-cbind(inputProspect,prospectLGBoost = prospectLGBoost)

#adding the LGBoost prediction to the entire training data
trainingLGBoost<-predict(model_logitboost,treatedmodel_d,type='prob')
trainingDT<-cbind(trainingDT,trainingLGBoost=trainingLGBoost)


#######################################################################################################################
#RUNNING KNN model
########################################################################################################################


# The caret package is robust and lets you apply the needed scaling during the fit
knnFit <- train(factor(Y_AcceptedOffer) ~ ., #similar formula to lm
                data = treatedTrain, #data input
                method = "knn", #caret has other methods so specify KNN
                preProcess = c("center","scale"),
                tuneGrid = expand.grid(k = c(15, 21, 25, 29,31,37,45))) #normalization

knnFit <- train(factor(Y_AcceptedOffer) ~ ., #similar formula to lm
                data = treatedTrain, #data input
                method = "knn", #caret has other methods so specify KNN
                preProcess = c("center","scale"),
                tuneLength=50) #normalization
# Evaluation
knnFit
plot(knnFit)


# training set accuracy
knncutoff<-0.5

trainClasses <- predict(knnFit,treatedTrain)
trainClasses <- ifelse(trainClasses >= knncutoff, 1,0)
resultsDF    <- data.frame(actual = treatedTrain$Y_AcceptedOffer, 
                           classes = trainClasses)
head(resultsDF)


table(trainClasses,treatedTrain$Y_AcceptedOffer) 

# Testing set accuracy; PREDICT WILL CENTER THE NEW DATA FOR YOU!!
testClasses <- predict(knnFit,treatedTest)
table(testClasses,treatedTest$Y_AcceptedOffer)

Accuracy(testClasses, treatedTest$Y_AcceptedOffer)

prospectKNN = predict(knnFit,treatedmodel_prospect,type='prob')

inputProspect<-cbind(inputProspect,prospectKNN = prospectKNN)
write.csv(inputProspect,file="predictedProspect.csv")

#adding the KNN predictions to the entire training data
trainingKNN<-predict(knnFit,treatedmodel_d,type='prob')
trainingDT<-cbind(trainingDT,trainingKNN=trainingKNN)


# To see probabilities 
trainProbs <- predict(knnFit, treatedTrain, type=c('prob'))
head(trainProbs)

# What is the column with the maximum value for each record? ONLY first 6 as example
topProb <- max.col(head(trainProbs))

# Get the name of the top valued probability
names(trainProbs)[topProb]

# Is this the same as predicting the classes directly?
head(as.character(trainClasses))






###############################Analyze####################################
#In this phase, I will analyze the results of the model's predictions
#Output of the model will be exported, to be shown to the Chief Product Officer
###############################Analyze####################################


finalreport<-left_join(inputProspect,model_prospect, by = c('HHuniqueID'))
finalreport$combinedscore <-finalreport$DTPrediction.1+finalreport$prospectRF500.1+finalreport$prospectRF100.1+finalreport$prospectRFtc.1+finalreport$prospectLGBoost.1+finalreport$prospectKNN.1
finalreport$rank<-NA
finalreport$rank<-rank(-finalreport$combinedscore)

trainingDT$combinedscore<-trainingDT$DTPrediction.1+trainingDT$RF500rediction.1+trainingDT$RF100rediction.1+trainingDT$trainingRFtc.1+trainingDT$trainingLG+trainingDT$trainingLGBoost.1+trainingDT$trainingKNN.1
trainingDT$rank<-NA
trainingDT$rank<-rank(-trainingDT$combinedscore)

write.csv(finalreport,file="predictedProspect.csv")
write.csv(trainingDT,file="trainingpredicted.csv")

#Exported the predictions for the 4000 training records, as well as the 1000 prospects into respective CSV files to look at.
#Analyzed and found that the the top 
#END


