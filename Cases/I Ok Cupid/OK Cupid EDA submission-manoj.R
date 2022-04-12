# Case-I OK Cupid - CSCI E-96 - Data Mining for Business
# 
# Purpose: Exploratory Data Analysis of OK Cupid profiles in order to 
# create micro segments and personas for future campaigns.
# Goal is to examine the data, clean it, use supplemntal data to enrich and
# identify 4 or more interesting insights from the user data. Findings from this analysis
# will be presented to the head of marketing, who is looking for an "ah-ha" persona or previously 
# unknown data relationship.
# 
# Student name : Manoj Sharma


#load the  libraries to be used in this script
library(tidyverse)
library(DataExplorer)
library(maps)
library(SmartEDA)
library(leaflet)
library(tm)
library(stringr)
library(psych)

options(scipen = 999) # to avoid the scientific notations
#LOADING THE DATA
#setting the working directory from which I am importing data
setwd("C:/Users/manoj/Harvard Courses/Harvard_DataMining_Business_Student/Cases/I Ok Cupid")

dfprofiles<-read_csv('profiles.csv') #Importing the profiles.csv data into dfprofiles dataframe
dflatlon<- read.csv('LatLon.csv') #Importing latitude and longitude data for the locations
dfaddr<- read.csv('addr.csv') # importing the addresses data for those locations 
dfcensus<-read_csv('sharedCensus2010Vars.csv') # importing the 2010 census data


#SAMPLING THE DATA

#Sampling data to see the type of data available in various columns
glimpse(dfprofiles)

describe(dfprofiles)


head(dfprofiles) #view top 5 rows from the profiles dataframe
tail(dfprofiles) #view bottom 5 rows from the profiles dataframe


#EXPLORING THE DATA



#plotting missing values from the profiles data to check data quality for each column
plot_missing(dfprofiles)

table(dfprofiles$education) #viewing various responses on the education column with no. of profiles

unique(dfprofiles$pets) # listing unique responses from the pets column

#Finding out various categories in 'education' field
unique(dfprofiles$education)

#Finding out various categories in 'body_type' field
unique(dfprofiles$body_type)


#Analyzing languages
unique(dfprofiles$speaks)


#MODIFYING THE DATA

#Processing Income field
#Adding a dummy variable to indicate if income is revealed or not, adjusting for NA
dfprofiles<-dfprofiles %>%
  mutate(income_revealed=ifelse(is.na(income),0,1))


#processing the education field
#defaulting no education to high-school, assuming the profile was created on the internet, using a computer
dfprofiles<-dfprofiles %>%
  mutate(education=ifelse(is.na(education),"high school",education))


#added combined categories and levels to add a level of ordinality in education
dfedlevels<-read.csv('edlevels.csv')
head(dfedlevels) #exploring the top 5 rows of the dfedlevels dataframe


#processing the pets field
#16 types of responses in 'pets'
# need to classify in fewer categories
# these newer categories can be :
# 1. dislikes pets - 'dislikes dogs and dislikes cats'
# 2. dislikes dogs -  'dislikes dogs','dislikes dogs and likes cats','dislike dogs and has cats'
# 3. dislikes cats - 'dislikes cats', 'likes dogs and dislikes cats'
# 4. likes pets - 'likes dogs and likes cats', 'has dogs and likes cats','has dogs and has cats','likes dogs and has cats'
# 5. likes dogs - 'likes dogs','has dogs'
# 6. likes cats - 'has cats','likes cats'
# 7. indifferent about pets - NA


dfprofiles<-dfprofiles%>%
  mutate(pets=ifelse(is.na(pets),"indifferent about pets",pets)) #handling NAs by adding a default value in case of NA

#RECODING various pet choices into 6 categories
#not the best way to recode pet_preference from pets, but this quick and dirty code still does the job
dfprofiles<-dfprofiles%>%
  mutate(pet_preference=ifelse(pets=='dislikes dogs and dislikes cats' ,'dislikes pets', 
                             ifelse(pets=='dislikes dogs' | pets =='dislikes dogs and likes cats' | pets=='dislike dogs and has cats','dislikes dogs',
                                      ifelse(pets=='dislikes cats' |pets== 'likes dogs and dislikes cats','dislikes cats',
                                             ifelse(pets=='likes dogs and likes cats' |pets== 'has dogs and likes cats'|pets=='has dogs and has cats'|pets=='likes dogs and has cats','likes pets',
                                                    ifelse(pets=='likes dogs' |pets=='has dogs','likes dogs',
                                                         ifelse(pets=='has cats'|pets=='likes cats','likes cats','indifferent about pets')))))))




#RECODING various body_type responses into 2 classifications - "conscious-concerned" or "unconcerned"
#not the best way to recode self body_image from body_type, but this quick and dirty code still does the job
dfprofiles<-dfprofiles%>%
  mutate(self_body_image=ifelse(body_type=='a little extra'|body_type=='full figured'|
                                  body_type=='thin'|body_type=='overweight'|body_type=='rather not say'|
                                  body_type=='used up'|body_type=='curvy'|body_type=='a little extra'|is.na(body_type) ,'conscious-concerned','unconcerned')) 
                               

#Splitting the 'speaks' field into speaks_first and speaks_second to identify first language, and other languages
dfprofiles[c('speaks_first','speaks_second')]<-str_split_fixed(dfprofiles$speaks,",",2)

#Extracting second language and cleaning the field
dfprofiles$speaks_second<-str_extract(dfprofiles$speaks_second,"(\\w+)") 

#Extracting first language and cleaning the field
dfprofiles$speaks_first<-str_extract(dfprofiles$speaks_first,"(\\w+)") 
#validating the output of the processing on speaks field and the split into first and second languages
table(dfprofiles$speaks_first)

#displaying the second language by frequency in a descending order
table(dfprofiles$speaks_second) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))


#performing a left_join to add latlon columns into the dfaddr dataset
dfaddr<-left_join(dfaddr,dflatlon,by='location')

dfaddr
#identifying all possible values in Education, entered by people
unique(dfprofiles$education)
#found a total of 33 unique items..

#recoding those 33 categories of education by performing a lookup type of operation using left join with dfedlevels field
#dfedlevels was imported from a manually created mapping from 33 to 5 levels which are : middle-school, high school, under grad, grad school 
#  and doctorate for education levels and also adding ordinality to this data
dfprofiles<-left_join(dfprofiles,dfedlevels,by='education')


hist(dfprofiles$age) #Visualizing no. of profiles by age

ggplot(dfprofiles, aes(x=pet_preference, fill=sex)) + 
  geom_histogram(stat="count")

ggplot(dfprofiles, aes(x=self_body_image, fill=sex)) + 
  geom_histogram(stat="count") 

table(dfprofiles$self_body_image,dfprofiles$sex)


#Identifying top second languages from the newly created field 'speaks_second, by sex
ggplot(dfprofiles, aes(x=speaks_second, fill=sex)) + 
  geom_histogram(stat="count")+
  xlim(names(sort(table(dfprofiles$speaks_second),decreasing=TRUE)[1:4]))


#Visualizing education levels by the newly created ed_category field
ggplot(dfprofiles, aes(x=ed_category)) +
  geom_bar(fill='steelblue')


#seeing the data of ed_category and no. profiles
table(dfprofiles$ed_category)

#drug use
table(dfprofiles$drugs)

#adding a dummy variable for drug usage
dfprofiles<-dfprofiles %>%
  mutate(drug_usage=ifelse(is.na(drugs),0,
                           ifelse(drugs=='never',0,
                              ifelse(drugs=='often',1,
                                  ifelse(drugs=='sometimes',1,0)))))
                                         
                                  
                           

#Visualization the no. of profiles by education field, which is hard to see and understand
ggplot(dfprofiles, aes(x=education)) +
  geom_bar(fill='steelblue') +
  labs(x='Education')


#adding ordinality into the ed_category by defining the factor and levels
dfprofiles$ed_category = factor(dfprofiles$ed_category, levels = c('middle school', 'high school', 'under grad','grad school','doctorate'))


#plotting the count of profiles by ed_category in the asending order of level of education
ggplot(dfprofiles, aes(x=ed_category)) +
  geom_bar(fill='steelblue') +
  labs(x='Team')

#plotting the count of profiles by ed_category and sex
ggplot(dfedbysex, aes(fill=sex, y=n, x=ed_category)) + 
  geom_bar(position="dodge", stat="identity") 



#rechecking for missing (NA) data after cleanup to validate the outcome
plot_missing(dfprofiles)
plot_histogram(dfprofiles)
plot_density(dfprofiles)


dfedbysex<-dfprofiles%>%
  count(sex,ed_category,smokes,sort=TRUE)

table(dfprofiles$ed_category,dfprofiles$sex)

dflocation_summary<-count(dfprofiles,location,name='num_profiles')


ggplot(dfedbysex, aes(fill=sex, y=n, x=ed_category)) + 
  geom_bar(position="dodge", stat="identity") +
  facet_wrap(~smokes)

dfedbysexdrugs<-dfprofiles%>%
  count(sex,ed_category,drug_usage,sort=TRUE)

ggplot(dfedbysexdrugs, aes(fill=drug_usage, y=n, x=ed_category)) + 
  geom_bar(position="dodge", stat="identity") +
  facet_wrap(~sex)

dfprofiles<-left_join(dfprofiles,dfaddr,by='location')


dfaddr<-left_join(dfaddr,dflocation_summary,by='location')
dfaddr





#Creating the maps and overlaying the location data on the map
map()
map('usa')
dev.off()
map( interior=T)
points(dfprofiles$lon,dfprofiles$lat,col='red')

glimpse(dfaddr)




#using MPLOT and leaflet function to create an interactive map of the profiles and locations
mplot<-leaflet(data=dfaddr) %>%
  addTiles() %>%
  addMarkers(popup = paste("Loc:",dfaddr$location,"<br>",
                           "Size:",dfaddr$num_profiles))

mplot

# first 20 quakes
df.20 <- dfaddr

getColor <- function(dfaddr) {
  sapply(dfaddr$num_profiles, function(num_profiles) {
    if(num_profiles <= 10) {
      "green"
    } else if(num_profiles <= 50) {
      "orange"
    } else {
      "red"
    } })
}

icons <- awesomeIcons(
  icon = 'ios-close',
  iconColor = 'black',
  library = 'ion',
  markerColor = getColor(df.20)
)

leaflet(dfaddr) %>% addTiles() %>%
  addAwesomeMarkers(~lon, ~lat, icon=icons, label=~as.character(num_profiles))

mplot

