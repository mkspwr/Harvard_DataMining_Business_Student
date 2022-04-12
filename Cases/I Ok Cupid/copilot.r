#import profiles.csv
#import LatLon.csv
#import addr.csv
library("tm")
library("SnowballC")
library("wordcloud2")
library("RColorBrewer")
library("dplyr")
library(wordcloud)
setwd("C:\\Users\\Manoj\\Harvard Courses\\Harvard_DataMining_Business_Student\\Cases\\I Ok Cupid")
profiles <- read.csv('profiles.csv')

#create a wordcloud of the top 10 most common words in the essay0 field
text<-profiles$essay0
docs<-Corpus(VectorSource(text))
docs<-docs%>%
    tm_map(removeNumbers)%>%
    tm_map(removePunctuation)%>%
    tm_map(stripWhitespace)
docs<-tm_map(docs,content_transformer(tolower))
docs<-tm_map(docs,removeWords,stopwords("english"))

#plot the wordcloud
wordcloud(docs,
          stopwords=stopwords("english"),
          random_state=1,
          max_words=10)


