library(Matrix)
library(dplyr)
library(data.table)
library(tidyverse)
library(DataExplorer)
library(vtreat)
library(MLmetrics)
library(car)
library(chron)
library(caret)
setwd("C:/Users/mksharma/Harvard/Harvard_DataMining_Business_Student")
setwd("./Cases/II National City Bank/training")
currentData   <- read.csv('CurrentCustomerMktgResults.csv')
vehicleData <- read.csv('householdVehicleData.csv') 
axiomData <- read.csv('householdAxiomData.csv')
creditData <- read.csv('householdCreditData.csv')


joinData <- left_join(currentData,vehicleData,by = c('HHuniqueID'))
joinData <- left_join(joinData, axiomData, by = c('HHuniqueID'))
joinData <- left_join(joinData, creditData, by = c('HHuniqueID'))
names(joinData)


plot_missing(joinData)
trainData<- joinData



clean_d <- trainData

max(clean_d[clean_d$LastContactMonth == 'feb', 'LastContactDay']) # check if the year the calls were made isn't a leap year (still could be, but awww well)
clean_d$DateCall <- as.Date(paste(clean_d$LastContactDay, clean_d$LastContactMonth, "2015", sep = '/'), "%d/%b/%Y") # 2015 wasn't a leap year, so that's the one we take
clean_d$Weekday <- factor(weekdays(clean_d$DateCall))

# Next, we want to know what time people were called during the day. 
# Let's see when the calls were made and what the working hours are.
plot(table(clean_d$CallStart)) # not very informative let's take the minutes and seconds off
plot(table(call_hr <- gsub("(:\\d{2})", "", clean_d$CallStart))) # ok... they are pretty diligent in calling people. Little dip just before/around lunch time

# We could take the times as they are given. However, that would be too much noise, in my opinion. Therefore, I opt for three time slots, i.e. morning 9 - 11:59:59, midday 12 - 14:59:59, afternoon the rest
clean_d$CallDayTime <- as.numeric(gsub("(:\\d{2})", "", clean_d$CallStart))

clean_d$CallDayTime <- factor(recode(clean_d$CallDayTime, "c('9', '10', '11')='morning'; c('12', '13', '14')='midday'; else='afternoon'"))

clean_d$call_dur_min <- 60 * 24 * as.numeric(times(clean_d$CallEnd)-times(clean_d$CallStart))

na_count <-sapply(clean_d, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)


summary(clean_d)
names(clean_d)
sub_clean_d <- subset(clean_d, select = -c(dataID, LastContactDay, LastContactMonth, past_Outcome, CallStart, CallEnd, DateCall))

names(sub_clean_d)
model_d <- sub_clean_d


table(model_d$carYr)
#function to find Mode, to impute Mode for NAs on car Year
Mode <- function(x) {
   ux <- unique(x)
   tab <- tabulate(match(x, ux))
   ux[tab == max(tab)]
}
Mode(model_d$carYr) #finding mode, turns out that mode in this case is 2019

train_control = trainControl(method="cv", number=10)

plot_missing(model_d)
model_d<-model_d %>%
  mutate(Education=ifelse(is.na(Education),"not provided",Education)) %>%
  mutate(Job=ifelse(is.na(Job),"not provided",Job)) %>%
  mutate(Communication=ifelse(is.na(Communication),"not provided",Communication)) %>%
  mutate(carYr=ifelse(is.na(carYr),2019,carYr))


plot_missing(model_d) #no missing data anymore
#columns need to be rearranged
names(model_d)
model_d = select(model_d,"HHuniqueID","Communication","NoOfContacts","DaysPassed","PrevAttempts",
"carMake","carModel",
"carYr","headOfhouseholdGender",   
"annualDonations","EstRace",
"PetsPurchases","DigitalHabits_5_AlwaysOn",
"AffluencePurchases","Age",
"Job","Marital",
"Education","DefaultOnRecord",
"RecentBalance","HHInsurance",
"CarLoan","Weekday",
"CallDayTime","call_dur_min","Y_AcceptedOffer")
names(model_d)
head(model_d)


# Apply to xVars
treatedX <- prepare(plan, model_d)
head(treatedX)


set.seed(42)
train_index <- createDataPartition(row_number(model_d), p = 0.75, list = FALSE, times = 1)
training <- model_d[train_index, ]
testing <- model_d[-train_index, ]

table(training$Y_AcceptedOffer)
table(testing$Y_AcceptedOffer)
# Cross validation - the train_control object will tell the model how to partition the data
train_control = trainControl(method="cv", number=10)

# Training the actual model. We have to pass CarInsurance, i.e. the variable to be classified, as a factor, not as a numeric. Otherwise the model gets confused and thinks we want to create a regression prediction instead of a binary classification
set.seed(42)
plot_missing(training)
summary(training)
model_rf = train(factor(Y_AcceptedOffer)~., data=training, trControl=train_control, method="rf")

# We make a frame and fill it with the predicted values. This allows us to the test the quality of the model in the next step
prediction_rf = predict(model_rf, subset(testing, select=-c(Y_AcceptedOffer)))
prediction_rf
testing$CarInsurance
#Compute the accuracy of predictions with a confusion matrix
confusionMatrix(prediction_rf, as.factor(testing$Y_AcceptedOffer))

#trying logistic regression
set.seed(42)
model_logreg <- glm(factor(factor(Y_AcceptedOffer)) ~., family=binomial(link='logit'), data=training)

prediction_logreg = predict(model_logreg, subset(testing, select=-c(factor(Y_AcceptedOffer))), type='response') # by choosing the type = response, we get the actual probabilities. Otherwise it would be the logodds

table(testing$Y_AcceptedOffer, prediction_logreg > 0.5) # LogReg gives the results as probabilities, so we can't use the confusionMatrix from caret. Does the same thing here

ggplot(results, aes(x=prediction_logreg, color=as.factor(actual))) +
  geom_density() + 
  geom_vline(aes(xintercept = cutoff), color = 'green')
  
set.seed(42)
model_logitboost <- train(factor(CarInsurance)~., data=training, trainControl=train_control, method="LogitBoost", nIter=50)

prediction_logitboost = predict(model_logitboost, subset(testing, select=-c(CarInsurance)))
confusionMatrix(prediction_logitboost,as.factor(testing$CarInsurance))


