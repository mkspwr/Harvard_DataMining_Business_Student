# Case-II OK Cupid - CSCI E-96 - Data Mining for Business
# For National City Bank use only
# Purpose: Create a customer propensity model for a new product, specifically a line of credit
# against a household's used car.
# Bank has historical data from 4000 previous calls and mailings for the line of credit offer. Using this data
# I prepared prediction models using various approaches to shortlist customer IDs from a list of 1000 
# prospective customers
# Student name : Manoj Sharma

#Loading R libraries for the analysis

library(ggplot2)
library(cowplot)
library(randomForest)
library(DataExplorer)
library(tidyverse)
library(caret)
library(vtreat)
library(chron)
library(MLmetrics)
library(rpart)
library(rpart.plot)
library(caretEnsemble)


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


#######################################################################################################
#Using SEMMA based approach..as taught in the class
#SEMMA - Sample - Explore - Modify- Model - Analyze
#SAMPLING DATA


head(joinData)
tail(joinData)
names(joinData)
names(prospectdata)



#Data exploration leads to a plan for treating
clean_prospect<-subset(prospectdata, select = -c(Age.y,Job.y,Marital.y,Education.y,HHInsurance.y,CarLoan.y,
                                                 Communication.y,LastContactDay.y,LastContactMonth.y, NoOfContacts.y, DaysPassed.y,
                                                 PrevAttempts.y, Balance,CarInsurance,DefaultOnRecord,Outcome))

names(clean_prospect)
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

plot(table(clean_d$CallStart)) # not very informative let's take the minutes and seconds off
plot(table(call_hr <- gsub("(:\\d{2})", "", clean_d$CallStart)))

# We could take the times as they are given. However, that would be too much noise, in my opinion. Therefore, I opt for three time slots, i.e. morning 9 - 11:59:59, midday 12 - 14:59:59, afternoon the rest
clean_d$CallDayTime <- as.numeric(gsub("(:\\d{2})", "", clean_d$CallStart))
clean_prospect$CallDayTime <- as.numeric(gsub("(:\\d{2})", "", clean_prospect$CallStart))


#clean_d$CallDayTime <- factor(recode(clean_d$CallDayTime, "c('9', '10', '11')='morning'; c('12', '13', '14')='midday'; else='afternoon'"))

clean_d$call_dur_min <- 60 * 24 * as.numeric(times(clean_d$CallEnd)-times(clean_d$CallStart))
clean_prospect$call_dur_min <- 60 * 24 * as.numeric(times(clean_prospect$CallEnd)-times(clean_prospect$CallStart))


#clean_d$Education <- factor(recode(clean_d$Education, "c('primary')=1; c('secondary')=2; c('tertiary')=3 ; else=1 "))
#clean_prospect$Education <- factor(recode(clean_prospect$Education.x, "c('primary')=1; c('secondary')=2; c('tertiary')=3 ; else=1 "))


#na_count <-sapply(clean_d, function(y) sum(length(which(is.na(y)))))
#na_count <- data.frame(na_count)


summary(clean_d)
names(clean_d)
names(clean_prospect)
sub_clean_d <- subset(clean_d, select = -c(dataID, LastContactDay, LastContactMonth,  CallStart, CallEnd, DateCall))

#removing the education variable which has been imputed earlier
sub_cleanProspect<-subset(clean_prospect, select = -c(dataID,LastContactDay.x,LastContactMonth.x,CallStart, CallEnd, DateCall ))
#Renaming the variables to be consistent

sub_cleanProspect<-
  rename( sub_cleanProspect
       , Job = Job.x
       , Age = Age.x
       , Marital = Marital.x
       , Communication = Communication.x
       , NoOfContacts = NoOfContacts.x
       , DaysPassed = DaysPassed.x
       , PrevAttempts = PrevAttempts.x
       , HHInsurance = HHInsurance.x
       , CarLoan = CarLoan.x
       , DefaultOnRecord = Default
       , Education = Education.x
      )

names(sub_clean_d)
names(sub_cleanProspect)
sub_clean_d = select(sub_clean_d, "HHuniqueID", "Communication", "NoOfContacts", "DaysPassed", "PrevAttempts", "past_Outcome"             
                           ,  "carMake", "carModel", "carYr", "headOfhouseholdGender", "annualDonations", "EstRace"
                           , "PetsPurchases", "DigitalHabits_5_AlwaysOn", "AffluencePurchases", "Age", "Job", "Marital", "Education"               
                           , "DefaultOnRecord", "RecentBalance", "HHInsurance", "CarLoan", "Weekday", "CallDayTime", "call_dur_min","Y_AcceptedOffer" )
sub_cleanProspect = select(sub_cleanProspect, "HHuniqueID", "Communication", "NoOfContacts", "DaysPassed", "PrevAttempts", "past_Outcome"             
                           ,  "carMake", "carModel", "carYr", "headOfhouseholdGender", "annualDonations", "EstRace"
                          , "PetsPurchases", "DigitalHabits_5_AlwaysOn", "AffluencePurchases", "Age", "Job", "Marital", "Education"               
                          , "DefaultOnRecord", "RecentBalance", "HHInsurance", "CarLoan", "Weekday", "CallDayTime", "call_dur_min","Y_AcceptedOffer" )


model_d <- sub_clean_d


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
  mutate(Y_AcceptedOffer=ifelse(Y_AcceptedOffer=="Accepted",1,0))%>%
  mutate(past_Outcome=ifelse(is.na(past_Outcome),"Not Available",past_Outcome)) %>%
  mutate(Education=ifelse(is.na(Education),"Not Available",Education))

model_d$Y_AcceptedOffer<-as.factor(model_d$Y_AcceptedOffer)
model_d$DefaultOnRecord<-as.factor(model_d$DefaultOnRecord)
model_d$HHInsurance<-as.factor(model_d$HHInsurance)
model_d$PetsPurchases<-as.factor(model_d$PetsPurchases)
model_d$PrevAttempts<-as.factor(model_d$PrevAttempts)
model_d$Communication<-as.factor(model_d$Communication)
model_d$AffluencePurchases<-as.factor(model_d$AffluencePurchases)

model_prospect<-sub_cleanProspect %>%
  mutate(Job=ifelse(is.na(Job),"not provided",Job)) %>%
  mutate(Communication=ifelse(is.na(Communication),"not provided",Communication)) %>%
  mutate(carYr=ifelse(is.na(carYr),mode_carYr,carYr)) %>%
  mutate(Y_AcceptedOffer=ifelse(Y_AcceptedOffer=="Accepted",1,0))%>%
  mutate(past_Outcome=ifelse(is.na(past_Outcome),"Not Available",past_Outcome)) %>%
  mutate(Education=ifelse(is.na(Education),"Not Available",Education))
  
model_prospect$Y_AcceptedOffer<-as.factor(model_prospect$Y_AcceptedOffer)
model_prospect$DefaultOnRecord<-as.factor(model_prospect$DefaultOnRecord)
model_prospect$HHInsurance<-as.factor(model_prospect$HHInsurance)
model_prospect$PetsPurchases<-as.factor(model_prospect$PetsPurchases)
model_prospect$PrevAttempts<-as.factor(model_prospect$PrevAttempts)
model_prospect$Communication<-as.factor(model_prospect$Communication)
model_prospect$AffluencePurchases<-as.factor(model_prospect$AffluencePurchases)


plot_missing(model_d) #no missing data anymore

plot_missing(model_prospect)


table(trainData$Marital)
table(model_prospect$Marital)


names(trainData)

#visualizing training data
ggplot(trainData,aes(x=headOfhouseholdGender,fill=Y_AcceptedOffer)) +
  geom_histogram(stat="count")

#visualizing prospects data, in order to make the model work, the data needs to be similar
ggplot(model_prospect,aes(x=headOfhouseholdGender)) +
  geom_histogram(stat="count")


ggplot(trainData,aes(x=Marital, fill=Y_AcceptedOffer)) +
  geom_histogram(stat="count")

ggplot(trainData,aes(x=carMake, fill=Y_AcceptedOffer)) +
  geom_histogram(stat="count")

ggplot(trainData,aes(x=age, fill=Y_AcceptedOffer)) +
  geom_histogram(stat="count")

ggplot(trainData,aes(x=Age,stat(density), colour=Y_AcceptedOffer)) +
  geom_freqpoly(binwidth=5)


names(model_d)
head(model_d)
plot_missing(model_d)
plot_missing(model_prospect)


names(model_d)
# Treatment
targetVar       <- names(model_d)[27]
informativeVars <- names(model_d)[2:26]

# Design a "C"ategorical variable plan 
plan <- designTreatmentsC(model_d, 
                          informativeVars,
                          targetVar,1)
#applying treatment to the training data
treatedmodel_d <- prepare(plan, model_d)
write.csv(treatedmodel_d,file="revisedinput.csv")
#applying the same treatment to the prospects data
treatedmodel_prospect <- prepare(plan, model_prospect)


set.seed(42)
train_index <- createDataPartition(treatedmodel_d$Y_AcceptedOffer, p = 0.85, list = FALSE, times = 1)
training <- treatedmodel_d[train_index, ]
testing <- treatedmodel_d[-train_index, ]
plot_missing(training)
plot_missing(testing)

table(training$Y_AcceptedOffer)
table(testing$Y_AcceptedOffer)

str(model_d)

plot_missing(model_d)




modelrandomforest <- randomForest(Y_AcceptedOffer ~ ., data=training, ntree=300, mytry=27, proximity=TRUE)

importance(modelrandomforest)
varImpPlot(modelrandomforest)

predprospect <-predict(modelrandomforest,treatedmodel_prospect,type='prob')

head(modelrandomforest$err.rate) 
tail(modelrandomforest$err.rate)

oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "0", "1"), each=nrow(modelrandomforest$err.rate)),
  Error=c(modelrandomforest$err.rate[,"OOB"], 
          modelrandomforest$err.rate[,"0"], 
          modelrandomforest$err.rate[,"1"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))


modelRF1000 <- randomForest(Y_AcceptedOffer ~ ., data=training,ntree=1000, proximity=TRUE)

head(modelRF1000$err.rate) 
tail(modelRF1000$err.rate)

oob.error.data <- data.frame(
  Trees=rep(1:nrow(modelrandomforest$err.rate), times=3),
  Type=rep(c("OOB", "0", "1"), each=nrow(modelrandomforest$err.rate)),
  Error=c(modelrandomforest$err.rate[,"OOB"], 
          modelrandomforest$err.rate[,"0"], 
          modelrandomforest$err.rate[,"1"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))


oob.values <- vector(length=50)
for(i in 1:50) {
  temp.model <- randomForest(Y_AcceptedOffer ~ ., data=training, mtry=i, ntree=300)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values
## find the minimum error
min(oob.values)
## find the optimal value for mtry...
which(oob.values == min(oob.values))
## create a model for proximities using the best value for mtry
modelRFtuned <- randomForest(Y_AcceptedOffer ~ ., 
                      data=training,
                      ntree=300, 
                      proximity=TRUE, 
                      mtry=which(oob.values == min(oob.values)))

modelRFtuned

### Now let's apply to the validation test set
predtest        <- predict(modelRFtuned, testing, type='prob')
predtrain        <- predict(modelRFtuned, training, type = 'prob')
predprospect <-predict(modelRFtuned,model_prospect,type='prob')

# Accuracy Comparison from MLmetrics
Accuracy(testing$Y_AcceptedOffer, predtest) #Accuracy of the model on Testing data
Accuracy(training$Y_AcceptedOffer,predtrain)#Accuracy of the model on training data



distance.matrix <- as.dist(1-modelRFtuned$proximity)
mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)

mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=training$Y_AcceptedOffer)


ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_text(aes(color=Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")



dotplot(results,metric = 'RMSE')

modelcorr <- modelCor(results)
ggcorrplot(modelcorr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlation between models", 
           ggtheme=theme_bw)

