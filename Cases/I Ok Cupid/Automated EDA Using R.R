#Automated EDA using R

library(DataExplorer)
library(tidyverse)

# Set WD
setwd("C:\\Users\\Manoj\\Harvard Courses\\Harvard_DataMining_Business_Student\\Cases\\I Ok Cupid")
profiles <- read.csv('profiles.csv')
create_report(profiles)

install.packages("SmartEDA")
library(SmartEDA)

ExpReport(profiles, op_file='smarteda.html')

install.packages("dlookr")
install.packages("Tex")

library(dlookr)
diagnose_web_report(profiles)


summary(profiles)


