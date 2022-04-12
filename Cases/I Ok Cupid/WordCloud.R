# Install
#install.packages("tm")  # for text mining
#install.packages("SnowballC") # for text stemming
#install.packages("wordcloud") # word-cloud generator 
#install.packages("RColorBrewer") # color palettes

# Load
library("tm")
library(dplyr)
library("SnowballC")
library("wordcloud2")
library("RColorBrewer")

# Set WD
setwd("C:\\Users\\Manoj\\Harvard Courses\\Harvard_DataMining_Business_Student\\Cases\\I Ok Cupid")
profiles <- read.csv('profiles.csv')


#profilesFemale<-subset(profiles, profiles$sex=='f')
profilesMale<-subset(profiles,profiles$sex=='m')

sample_size=floor(0.6*nrow(profilesMale))
set.seed(1234)
profilesMale<-sample(seq_len(nrow(profilesMale)),size=sample_size)

profiles<-profiles[profilesMale,]

#summary(profilesFemale)
summary(profilesMale)


#Female_essays<-profilesFemale$essay0

male_essays<-profiles$essay0

#head(Female_essays)
# Load the data as a corpus for female profile essays
#fdocs <- Corpus(VectorSource(Female_essays))
mdocs <- Corpus(VectorSource(male_essays))


#inspect(fdocs)
inspect(mdocs)

# Convert the text to lower case
#fdocs <- tm_map(fdocs, content_transformer(tolower))
mdocs<-tm_map(mdocs,content_transformer(tolower))

# Remove numbers
#fdocs <- tm_map(fdocs, removeNumbers)
mdocs <- tm_map(mdocs, removeNumbers)

# Remove english common stopwords
#fdocs <- tm_map(fdocs, removeWords, stopwords("english"))
mdocs <- tm_map(mdocs, removeWords, stopwords("english"))

# Remove your own stop word
# specify your stopwords as a character vector
#fdocs <- tm_map(fdocs, removeWords, c("blabla1", "blabla2")) 
mdocs <- tm_map(mdocs, removeWords, c("blabla1", "blabla2")) 


# Remove punctuations
#fdocs <- tm_map(fdocs, removePunctuation)
mdocs <- tm_map(mdocs, removePunctuation)


# Eliminate extra white spaces
#fdocs <- tm_map(fdocs, stripWhitespace)
mdocs <- tm_map(mdocs, stripWhitespace)


# Text stemming
# docs <- tm_map(docs, stemDocument)


#fdtm <- TermDocumentMatrix(fdocs)
mdtm  <- TermDocumentMatrix(mdocs)

#fm <- as.matrix(fdtm)
mm <- as.matrix(mdtm)
  
  
#fv <- sort(rowSums(fm),decreasing=TRUE)
mv<- sort(rowSums(mm),decreasing=TRUE)
  
  
#fd <- data.frame(word = names(fv),freq=fv)
md <- data.frame(word = names(mv),freq=mv)


#head(fd, 10)
head(md,10)

set.seed(1234)
#wordcloud(words = fd$word, freq = fd$freq, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35,  colors=brewer.pal(8, "Dark2"))
wordcloud(words = md$word, freq = md$freq, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35,  colors=brewer.pal(8, "Dark2"))


#wordcloud2(fd,
#           size=0.5,
#           shape='circle',
#)


wordcloud2(md,
           size=0.5,
           shape='circle',
)




wordcloud2(fd,
           size=0.8,
           shape='star',
           rotateRatio  =0.5,
           minSize=1
)


#sentiment analysis
library(syuzhet)
library(lubridate)
library(ggplot2)
library(scales)
library(reshape2)
library(dplyr)


s<-get_nrc_sentiment(male_essays)
s_male<-s
s_female<-get_nrc_sentiment(Female_essays)

head(s)

#s_merged<-merge(s_male,s_female)

barplot(colSums(s_male),
        las=2,
        col=rainbow(10),
        ylab='Count',
        main='Sentiment Scores for Male essays')

barplot(colSums(s_female),
        las=2,
        col=rainbow(10),
        ylab='Count',
        main='Sentiment Scores for Female essays')

head(s_female)
