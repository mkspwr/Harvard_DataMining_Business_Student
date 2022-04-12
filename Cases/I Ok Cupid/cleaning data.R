#Cleaning bad data
#MCAR - data missing completely at random - 
#       - no relationship between missing data and circumstances, 
#       like order of rows, location etc..
#       - very rare, can be ignored
#MAR - data missing at random
#     - underlying circumstances which explain missing data
#     - this can be usually ignored

#MNAR - data missing not at random
#     - values of missing information, related to the reason
#     - eg. BP data missing, because either too high to read..
#     - serious concern


#load the tidyverse packages
library(tidyverse)
library(SmartEDA)

setwd("C:/Users/manoj/Harvard Courses/Harvard_DataMining_Business_Student/Cases/I Ok Cupid")
profiles<-read_csv('profiles.csv')

#add a column to profiles dataframe which has 0 for missing income and 1 for not missing income
profiles$income_missing<-profiles$income=='?'

profiles %>%
    mutate(income_missing=ifelse(income>0,1,0))


ExpNumStat(profiles,by="A",gp=NULL,Qnt=seq(0,1,0.1),MesofShape=2,Outlier=TRUE,round=2)


#create a wordcloud of the top 10 most common words in the essay0 field

setwd("C:/Users/manoj/Harvard Courses/Harvard_DataMining_Business_Student/Cases/I Ok Cupid")

profiles<-read_csv('profiles.csv')

text<-profiles$essay0
docs<-Corpus(VectorSource(text))
docs<-docs%>%
    tm_map(removeNumbers)%>%
    tm_map(removePunctuation)%>%
    tm_map(stripWhitespace)
docs<-tm_map(docs,content_transformer(tolower))
docs<-tm_map(docs,removeWords,stopwords("english"))

