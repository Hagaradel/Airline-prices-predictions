import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from preprocess_1 import *
#########################################################
# Multi_regeneration
data = pd.read_csv('airline-price-prediction.csv')

data['date'] = data['date'].replace('[-]', '/', regex=True)
data['price'] = data['price'].replace('[,]', '', regex=True).astype(int)
# There no null value
data.info()
# Features
X = data.iloc[:, :10]
# Label
Y = data['price']

# Pre_Processing
# read the text feature from the data
cols = ('ch_code', 'type', 'airline', 'stop','route')
feature_encoder(X, cols)

# deal with date by splitting  it into day, Month and year
date_cols = 'date'
date_handel(X, date_cols)
# X = X.drop(columns=['date'])
###############################################
# deal with time dep_time & arr_time
time_cols = 'dep_time'
time_cols2 = 'arr_time'
time_handel(X, time_cols)
time_handel2(X, time_cols2)
time1 = 'new_dep_time'
time2 = 'new_arr_time'
time_taken(X, time1, time2)
# X = X.drop(columns=['dep_time', 'arr_time', 'time_taken', 'date_year'])
X.info()

################################################
new_data = X
new_data['price'] = Y
######################
# corr = new_data.corr()
# plt.subplots(figsize=(12, 8))
# sns.heatmap(corr, annot=True)
# plt.show()
################################################
# Feature Selection
# Get the correlation between the features
corr = new_data.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['price']) >= 0.1]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = new_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
X = X[top_feature]
##############################################
# feature scaling
X = feature_scaling(X)
###############################
# best multi_regeneration with 5 features airline, ch_code, num_code, stop, type
# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
# multi_regeneration 1
cls = linear_model.LinearRegression()
cls.fit(X_train, y_train)
prediction = cls.predict(X_test)

print('Co-efficient of linear regression', cls.coef_)
print('Intercept of linear regression model', cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

#########################################################
