#EDA for OK Cupid data

# 7 packages to add
library(tidyverse)
library(choroplethr)
library(choroplethrMaps)
library(openintro)

#library(fiftystater)
#library(colorplaner)


#Importing Profiles data

# Set WD
setwd("C:\\Users\\Manoj\\Harvard Courses\\Harvard_DataMining_Business_Student\\Cases\\I Ok Cupid")

# See all files in wd, leave this blank inside the parentheses
dir()

# Get the okcupid data as `profiles`
profiles <- read.csv('profiles.csv')
latlon<- read.csv('LatLon.csv')
addr<- read.csv('addr.csv')

#Converting the dataset into data frame tibbles
dfprofiles<-as_data_frame(profiles)
dflatlon<-as_data_frame(latlon)
dfaddr<-as_data_frame(addr)

head(dflatlon)
head(dfaddr)

dfaddr<-left_join(dfaddr,dflatlon,by='location')
dfaddr

#replac


# rural vs urban for targetting
# last login vs education vs job vs income etc etc
#add dummy variables for education, sex, orientation, religion, diet



#Visualization
#Histogram
profiles%>%
  ggplot(aes(x=age,fill=sex)) +
  geom_histogram(alpha=0.8,color='darkblue') +
  ggtitle('Profiles by age and sex') +
  facet_wrap(~sex)

#Density
profiles%>%
  ggplot(aes(x=age,fill=sex)) +
  geom_density(alpha=0.8,color='darkblue') +
  ggtitle('Profiles by age and sex') 

#Scatter
profiles%>%
#  filter(sex=='m' | sex=='f')%>%
  ggplot(aes(x=age,y=income, col=sex )) +
  geom_point(alpha=0.5) +
  ggtitle('Profiles by age and sex') 

