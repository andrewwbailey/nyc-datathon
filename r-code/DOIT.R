library(bigrquery)
library(Quandl)

project <- "datathon-1251" # put your projectID here

jobs_query <- 'SELECT * from dataset.jobs limit 100;'
categories_query <- 'SELECT * from dataset.categories limit 100;'
companies_query <- 'SELECT * from dataset.companies limit 100;'
companies_daily_query <- 'SELECT * from dataset.companies_daily limit 100;'

jobs.data <- query_exec(jobs_query,project)
categories.data <- query_exec(categories_query,project)
companies.data <- query_exec(companies_query,project)
#companies_daily.data <- query_exec(companies_daily_query,project)

gdp<-Quandl("FRED/NGDPPOT", start_date="2007-08-01", end_date="2015-12-31")
write.table(gdp, file="GDP_quart.csv", quote=FALSE)


####################Calculate slopes/growth from job postings##################################
setwd("~/Documents/Correlation1/Data_Jobs")
files<-list.files(pattern=".csv")

new<-matrix(data=NA, nrow=0, ncol=4)
colnames(new)<-c("jobID","slope","R","p")
for (i in 1:length(files)){
  assign(files[i], read.table(files[i]))
  a<-order(get(files[i])[,2])
  temp<-get(files[i])[a,]
  
  
  x<- c(1:dim(temp)[1])
  y<- temp[,2]
  
  sum<- summary(lm(x ~ y))
  
  p <- sum$coefficients[2,4] 
  r <- sum$r.squared
  slope <- sum$coefficients[2,1]
  
  new<- rbind(new, c(files[i], slope, r, p))
  
}
write.table(new, file="Growth_per_category.csv",sep=",", quote=F)

################Ordered tables####################

for (i in 1:length(files)) {
  a<-order(get(files[i])[,2])
  temp<-get(files[i])[a,]
  write.table(temp, file=paste(files[i],".ordered"), sep=",", quote=F)
}


#############Merge Tables

temp<-merge(get(files[1]),get(files[2]),by="created_date")

temp<-merge(temp,get(files[3]), by="created_date")
temp<-merge(temp,get(files[4]), by="created_date")
temp<-merge(temp,get(files[5]), by="created_date")
temp<-merge(temp,get(files[6]), by="created_date")
temp<-merge(temp,get(files[7]), by="created_date")
temp<-merge(temp,get(files[8]), by="created_date")
temp<-merge(temp,get(files[9]), by="created_date")
temp<-merge(temp,get(files[10]), by="created_date")
temp<-merge(temp,get(files[11]), by="created_date")

colnames(temp)<-c("created_date", files)
write.table(temp,file="Jobs_ordered_merges.txt", sep=",",quote=F)


############################
Predict <- read.csv("~/Documents/Correlation1/Predict.csv")
a<-order(Predict[,1])
Predict<- Predict[a,]
pdf(file="Predict vs Observed.pdf")
plot(Predict[,2]/Predict[,1], pch=20, ylab="Obs./Exp.", ylim=c(-1,2))
abline(h=1, col="red")
dev.off()

