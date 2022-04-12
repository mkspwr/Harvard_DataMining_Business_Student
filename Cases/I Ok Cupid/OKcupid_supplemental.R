#' Author: Ted Kwartler
#' Date: 2-14-2019
#' Purpose: OKCupid Case Supplemental
#' 
#' 

# Libs
#library(okcupiddata)
#' Problem statement
#' Examine Data, Clean Data, Supplemantal Data (optional), Analyze, Draw Conclusion/Insights
#' Gaps/Ethical Concerns(Optional)
library(dplyr)
library(ggplot2)
library(wordcloud2)
library(RColorBrewer)
library(tm)

# Set WD
setwd("C:\\Users\\Manoj\\Harvard Courses\\Harvard_DataMining_Business_Student\\Cases\\I Ok Cupid")

# See all files in wd, leave this blank inside the parentheses
dir()

# Get the okcupid data as `profiles`
profiles <- read.csv('profiles.csv')
latlon<- read.csv('LatLon.csv')
addr<- read.csv('addr.csv')

#setting the seed to retrieve same results under each iteration (reproducibility)
set.seed((1234))


#generate a word cloud of the top 10 most common words in the essay0 field
text<-profiles$essay0
docs<-Corpus(VectorSource(text))
docs<-docs%>%
tm_map(removeNumbers)%>%
tm_map(removePunctuation)%>%
tm_map(stripWhitespace)
docs<-tm_map(docs,content_transformer(tolower))
docs<-tm_map(docs,removeWords,stopwords("english"))

dtm<-TermDocumentMatrix(docs)
matrix<-as.matrix(dtm)
head(matrix)
words<-sort(rowSums(matrix),descending=TRUE)
df<-data.frame(word=names(words),freq=words)


wordcloud(words=df$word,freq=df$freq,min.freq=1,max.words=100,random.order=FALSE,rot.per=0.35,colors=brewer.pal(8,'Dark2'))


#taking a sample of 10% of the total data, for exploration
sampled.profile<-profiles[sample(nrow(profiles),size=0.10*nrow(profiles),replace=F),]
#summarizing the variables and types of each variable
summary(sampled.profile)

head(sampled.profile)

sampled.profile %>%
  select(c('age', 'body_type','ethnicity','height','sex')) %>%
  head(10) 

#setting the last online as a date format, in order to be able to do analysis
profiles$last_online<-as.Date(profiles$last_online)


##### I would do some basic EDA and plotting of individual vars then move to more complex interactions
table(profiles$orientation)
hist(profiles$age)

#plot a histogram of profiles by status
hist(profiles$status)

head(profiles)

df<-profiles[]
ggplot(df,aes(x=status))+geom_histogram()

##### Example 2 way EDA
table(profiles$age,profiles$orientation)


#### Missing in income & quick mean imputation example; you can still use vtreat instead to clean all this data but we are only exploring not modeling so maybe dont do it for this case.
sum(is.na(profiles$income))
profiles$income[is.na(profiles$income)] <- mean(profiles$income, na.rm=TRUE)

##### Feature Engineer relationship status & education if you thought there was a connection
profiles$statEDU <- paste(profiles$status, profiles$education, sep = '_')
table(profiles$statEDU)

##### Enrich with one of the new data sets, you may want to do this with the other csv
moreData <- left_join(profiles, latlon, by ='location')
head(moreData)

#### You can use complete.cases() to identify records without NA if that is the route you want to explore.  Of course you can use a function covered in class to visualize the variables with the hightest % of NA so you could drop those instead of all rows with an NA.
completeMoreData <- moreData[complete.cases(moreData),]

# End
profilesMatrix <- data.matrix(profiles)

heatmap(profilesMatrix,Rowv=NA, Colv=NA, scale="column")


