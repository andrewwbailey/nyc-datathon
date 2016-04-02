
#necessary libraries
import pandas as pd
import csv
from pandas.io import gbq

#jobs_query = """SELECT created_date, COUNT(*) FROM dataset.jobs GROUP BY created_date"""
#jobs_query = """SELECT trc_business_sector, COUNT(*) FROM dataset.companies GROUP BY trc_business_sector"""
#jobs_query = """SELECT trc_economic_sector, COUNT(*) FROM dataset.companies WHERE trc_economic_sector != 'None' GROUP BY trc_economic_sector"""
jobs_query = """SELECT created_date, COUNT(*) FROM dataset.jobs LEFT OUTER JOIN dataset.companies ON dataset.jobs.company_id = dataset.companies.id \
WHERE dataset.companies.trc_economic_sector = 'Cyclical Consumer Goods & Services' GROUP BY created_date"""
#other tables available include 'dataset.categories','dataset.companies','dataset.companies_daily'

project_id='datathon-1251'


# will be asked to authenticate
# credentials:
# Username: datathon01@gmail.com
# password: correlation2

# dataframe from the query output
df = gbq.read_gbq(jobs_query, project_id,reauth=False)

df.head()

with open("sectorJobsAdded.csv", 'a') as R_export:
	#file1 = csv.writer(R_export, delimiter='\t')
	#file1.writerow(df)
	df.to_csv(R_export, sep='\t')

#print df
