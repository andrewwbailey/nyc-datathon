```
The below script connects to a sample test dataset stored in Google BigQuery
and runs a select query
```

```
Package Requirements:

pip install google-api-python-client
pip install --upgrade pandas


Tested on version python 2.7

```

#necessary libraries
import pandas as pd
from pandas.io import gbq

jobs_query = """SELECT * from dataset.jobs limit 5"""
#other tables available include 'dataset.categories','dataset.companies','dataset.companies_daily'

project_id='datathonsample'


# will be asked to authenticate
# credentials:
# Username: datathon01@gmail.com
# password: correlation2

# dataframe from the query output
df = gbq.read_gbq(jobs_query, project_id,reauth=True)

df.head()
