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
# polynomial regression
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
# best polynomial regression with 5 features airline, ch_code, num_code, stop, type
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
poly_features = PolynomialFeatures(degree=3)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred = poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))


print('Co-efficient of linear regression', poly_model.coef_)
print('Intercept of linear regression model', poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
# Calculation of Accuracy
print(r2_score(y_test, ypred))
####################################################
