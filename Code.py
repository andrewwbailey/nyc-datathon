# -*- coding: utf-8 -*-
import pandas as pd
from pandas import isnull
import numpy as np
import matplotlib.pyplot as plt
#import RandomForestRegressor
import linear_model
import svm, neighbors
import NearestNeighbors
import KMeans
import LogisticRegression
import cross_validation
import accuracy_score
import roc_curve, auc
import average_precision_score
import scipy.stats as stats
import statsmodels.api as sm



#Reading file and indexing it with dates.
data = pd.read_csv('./data/stock_returns_base150.csv', index_col ='date', parse_dates=True);
columns_name = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
data.columns = columns_name;

#Check for normal distribution
stats.normaltest(data[1:100])

#Check for blank cells
isnull(data[0:100])

#Training data
S1 = data["S1"][0:50]
S2 = data["S2"][0:50]
S3 = data["S3"][0:50]
S4 = data["S4"][0:50]
S5 = data["S5"][0:50]
S6 = data["S6"][0:50]
S7 = data["S7"][0:50]
S8 = data["S8"][0:50]
S9 = data["S9"][0:50]
S10 = data["S10"][0:50]



#Plots showing how each of the variable is correlated with S1
plt.scatter(S1, S2)
plt.xlabel('S1')
plt.ylabel('S2')



plt.scatter(S1, S3)
plt.xlabel('S1')
plt.ylabel('S3')



plt.scatter(S1, S4)
plt.xlabel('S1')
plt.ylabel('S4')



plt.scatter(S1, S5)
plt.xlabel('S1')
plt.ylabel('S5')



plt.scatter(S1, S6)
plt.xlabel('S1')
plt.ylabel('S6')



plt.scatter(S1, S7)
plt.xlabel('S1')
plt.ylabel('S7')



plt.scatter(S1, S8)
plt.xlabel('S1')
plt.ylabel('S8')



plt.scatter(S1, S9)
plt.xlabel('S1')
plt.ylabel('S9')



plt.scatter(S1, S10)
plt.xlabel('S1')
plt.ylabel('S10')



#Plotting S1 to check for cyclic behavior
S1.plot(figsize=(16, 12))



#Spearman Rank Correlation and their p-values for feature selection
r_S2 = stats.spearmanr(S1, S2)
print 'Spearman Rank Coefficient: ', r_S2[0]
print 'p-value: ', r_S2[1]

r_S3 = stats.spearmanr(S1, S3)
print 'Spearman Rank Coefficient: ', r_S3[0]
print 'p-value: ', r_S3[1]

r_S4 = stats.spearmanr(S1, S4)
print 'Spearman Rank Coefficient: ', r_S4[0]
print 'p-value: ', r_S4[1]


r_S5 = stats.spearmanr(S1, S5)
print 'Spearman Rank Coefficient: ', r_S5[0]
print 'p-value: ', r_S5[1]

r_S6 = stats.spearmanr(S1, S6)
print 'Spearman Rank Coefficient: ', r_S6[0]
print 'p-value: ', r_S6[1]


r_S7 = stats.spearmanr(S1, S7)
print 'Spearman Rank Coefficient: ', r_S7[0]
print 'p-value: ', r_S7[1]


r_S8 = stats.spearmanr(S1, S8)
print 'Spearman Rank Coefficient: ', r_S8[0]
print 'p-value: ', r_S8[1]


r_S9 = stats.spearmanr(S1, S9)
print 'Spearman Rank Coefficient: ', r_S9[0]
print 'p-value: ', r_S9[1]


r_S10 = stats.spearmanr(S1, S10)
print 'Spearman Rank Coefficient: ', r_S10[0]
print 'p-value: ', r_S10[1]


#Rejecting S8 and S10 because of moderate correlation (0.5 Spearman rank coefficient).
feature_names = ["S2", "S3", "S4", "S5", "S6", "S7", "S9"]

#Training and Test data set
y_train = data["S1"][0:50]
x_train = data[feature_names][0:50]
x_test = data[feature_names][50:100]
#Exponential Moving Average of S1
ewma = pd.stats.moments.ewma
EMOV_n = ewma( y_train[1:50], com=5 )
X_train = np.array(x_train,EMOV_n)


#Modeling OLS
clf = linear_model.LinearRegression()
clf.fit ( X_train, y_train )
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Prediction
y_pre = clf.predict(x_test)
y_pre1 = clf.predict(X_train)
'''
print "Predictions based of OLS:"
print y_pre
'''

#Evaluation
'''
print ("Training R^2 coefficient: %0.2f" % clf.score(X_train, y_pre1))
print ("Testing R^2 coefficient: %0.2f" % clf.score(x_test, y_pre))
'''
diff = [abs(y_train[x] - y_pre1[x]) for x in range(len(y_pre1))]
s = sum(diff)/len(y_pre1)
print ("Sum of Absolute Deviations for training set using OLS: %0.2f" %s)


'''
#RandomForest Regressor based modeling
clf = RandomForestRegressor()
clf.fit(X_train, y_train)
scores = cross_validation.cross_val_score(clf, X_train, y_pre1, cv=5)
print("Accuracy after CV using Random Forest model: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Prediction
y_pre = clf.predict(x_test)
y_pre1 = clf.predict(X_train)
print "Predictions based of Random Forest:"
print y_pre

#Evaluation
print ("Training R^2 coefficient: %0.2f" % clf.score(X_train, y_pre1))
print ("Testing R^2 coefficient: %0.2f" % clf.score(x_test, y_pre))
diff = [abs(y_train[x] - y_pre1[x]) for x in range(len(y_pre1))]
s = sum(diff)/len(y_pre1)
print ("Sum of Absolute Deviations for training set using Random Forest : %0.2f" %s)
'''


#Output df
df = pd.DataFrame({'Date': data[50:100].index,
              'Value': y_pre[0:50]})

newdf = df[['Date', 'Value']].set_index('Date')
S1_final = df['Value'][0:50]
Date = df['Date']
newdf.plot()
plt.show()
#Writing output to CSV file
df.to_csv('./data/Predictions.csv', sep=',', index=False)











#model = sm.tsa.ARMA(x_train)

'''
df = pd.DataFrame({'S1': S1,
              'S2': S2,
              'S3': S3,
              'S4': S4,
			  'S5': S5,
			  'S6': S6,
			  'S7': S7,
			  'S9': S9})
'''

'''
OLS_model = regression.linear_model.OLS(S1, df[['S2','S3','S4','S5','S6','S7','S9']])
fitted_model = OLS_model.fit()
print 'p-value', fitted_model.f_pvalue
#print fitted_model.params

b_S2 = fitted_model.params['S2']
b_S3 = fitted_model.params['S3']
b_S4 = fitted_model.params['S4']
b_S5 = fitted_model.params['S5']
b_S6 = fitted_model.params['S6']
b_S7 = fitted_model.params['S7']
b_S9 = fitted_model.params['S9']

#S1 = data[52:100]["S1"]
S2 = data[52:100]["S2"]
S3 = data[52:100]["S3"]
S4 = data[52:100]["S4"]
S5 = data[52:100]["S5"]
S6 = data[52:100]["S6"]
S7 = data[52:100]["S7"]
S8 = data[52:100]["S8"]
S9 = data[52:100]["S9"]
S10 = data[52:100]["S10"]

predictions = b_S2 * S2 + b_S3 * S3 + b_S4 * S4 + b_S5 * S5 + b_S6 * S6 + b_S7 * S7 + b_S9 * S9
print predictions
'''


'''
model = pd.stats.ols.MovingOLS(y = y_train, x= x_train,
                             window_type='rolling',
                             window=30)
rolling_parameter_estimates = model.beta
rolling_parameter_estimates.plot();
plt.show()
'''

#arma_mod20 = sm.tsa.ARMA(data[0:50], (2,0), start=5/30/2014).fit()
#print arma_mod20.params
'''
model = sm.tsa.ARIMA(data[0:100].iloc[1:], order=(10, 0, 0))
results = model.fit()
print results.fittedvalues
'''
