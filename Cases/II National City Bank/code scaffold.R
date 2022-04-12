#' Case II Supplemental
#' TK
#' 4-30

# Libs
library(dplyr)
library(vtreat)
library(caret)
library(DataExplorer)
library(MLmetrics)
library(pROC)
library(ggplot2)
# Wd
setwd("./Cases/II National City Bank/training")

# Raw data, need to add others
currentData   <- read.csv('CurrentCustomerMktgResults.csv')
vehicleData <- read.csv('householdVehicleData.csv') 
axiomData <- read.csv('householdAxiomData.csv')
creditData <- read.csv('householdCreditData.csv')
trainData <- read.csv('carInsurance_train.csv')

head(currentData)
head(trainData)
# Perform sta join, neeed to add other data sets
joinData <- left_join(currentData, vehicleData, by = c('HHuniqueID'))
joinData <- left_join(joinData, axiomData, by = c('HHuniqueID'))
joinData <- left_join(joinData, creditData, by = c('HHuniqueID'))

joinData <- left_join(joinData, trainData, by = c('dataID'='Id'))

names(joinData)
# Let's drop ApplicationID,AgeBracket, Gender and raw text Summary
drops <- c("Age.y","Job.y","Marital.y","Education.y","HHInsurance.y","CarLoan.y","Communication.y",
           "LastContactDay.y","LastContactMonth.y","NoOfContacts.y", "DaysPassed.y","PrevAttempts.y",
           "CallStart.y","CallEnd.y","past_Outcome","Y_AcceptedOffer")

cleanedData<- joinData[, !(names(joinData) %in% drops)]
plot_missing(cleanedData)

names(cleanedData)
targetVar       <- names(cleanedData)[31]
informativeVars <- names(cleanedData)[3:30]

targetVar
plan <- designTreatmentsC(cleanedData, 
                          informativeVars,
                          targetVar,  1)

# Apply to xVars
treatedX <- prepare(plan, cleanedData)
head(treatedX)

fit <- glm(CarInsurance ~., data = treatedX, family ='binomial')
summary(fit)

# Backward Variable selection to reduce chances of multi-colinearity
# See chap6 for an explanation
# Takes 1m  to run so load a pre-saved copy that I already made 
bestFit <- step(fit, direction='backward')
saveRDS(bestFit, 'bestFit.rds')
bestFit <- readRDS('bestFit.rds')
summary(bestFit)

# Compare model size
length(coefficients(fit))
length(coefficients(bestFit))

# Get predictions
teamPreds <- predict(bestFit,  treatedX, type='response')
tail(teamPreds)

# Classify 
cutoff      <- 0.5
teamClasses <- ifelse(teamPreds >= cutoff, 1,0)

teamPreds
# Organize w/Actual
results <- data.frame(actual  = treatedX$CarInsurance,
                      recentBalance    = treatedX$RecentBalance,
                      seed    = treatedX$Age_x,
                      classes = teamClasses,
                      probs   = teamPreds)
head(results)

# Get a confusion matrix
(confMat <- ConfusionMatrix(results$classes, results$actual))

# Go to pptx slide 15

# What is the accuracy?
sum(diag(confMat)) / sum(confMat)

# This is the actual KPI Accuracy not to be confused with the forecast package function accuracy() which the book uses :(
Accuracy(results$classes, results$actual)

# Visually how well did we separate our classes?
ggplot(results, aes(x=probs, color=as.factor(actual))) +
  geom_density() + 
  geom_vline(aes(xintercept = cutoff), color = 'green')



#drop columns - 










head(joinData)
# This is a classification problem so ensure R knows Y isn't 0/1 as integers
joinData$Y_AcceptedOffer <- as.factor(joinData$Y_AcceptedOffer)
head(joinData)
boxplot(trainData)

## SAMPLE: Partition schema
set.seed(1234)
idx       <- ...
trainData <- ...
testData  <- ...

## EXPLORE: EDA, perform your EDA

## MODIFY: Vtreat, need to declare xVars & name of Y var
xVars <- c('DaysPassed', 'Communication', 'Outcome', ...)
yVar  <- '...'
plan  <- designTreatmentsC(..., xVars, ..., 1)

# Apply the rules to the set
treatedTrain <- prepare(..., trainData)
treatedTest  <- prepare(plan, ...)

## MODEL: caret etc.
fit <- train(Y_AcceptedOffer ~., data = ..., method = ...)

## ASSESS: Predict & calculate the KPI appropriate for classification
trainingPreds <- predict(..., ...)
testingPreds  <- predict(..., ...)

## NOW TO GET PROSPECTIVE CUSTOMER RESULTS
# 1. Load Raw Data
prospects <- read.csv('/cloud/project/cases/National City Bank/ProspectiveCustomers.csv')

# 2. Join with external data

# 3. Apply a treatment plan

# 4. Make predictions
prospectPreds <- predict(..., treatedProspects, type= 'prob')

# 5. Join probabilities back to ID
prospectsResults <- cbind(prospects$HHuniqueID, ...)

# 6. Identify the top 100 "success" class probabilities from prospectsResults


# End
