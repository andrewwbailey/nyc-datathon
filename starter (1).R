
#The below script connects to a sample test dataset stored in Google BigQuery
#and runs a select query


#necessary packages
install.packages('assertthat')
install.packages('bigrquery')
install.packages("httr")
install.packages("httpuv")

library(bigrquery)

# tested on version 3.2.3

#query include schema and table
jobs_query <- 'SELECT * from dataset.jobs limit 5;'
#other tables available include 'dataset.categories','dataset.companies','dataset.companies_daily'


project_id='datathonsample'

# will be asked to authenticate
# credentials:
# Username: datathon01@gmail.com
# password: correlation2

# if asked to save your auth token, select (1)
#the first time you will be asked if yu want to save your oauth token
data <- query_exec(jobs_query,project_id)