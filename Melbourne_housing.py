#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 22:57:53 2019

@author: deepti
"""

import pandas as pd
df = pd.read_csv('/home/deepti/Downloads/Melbourne/Melbourne_housing_dataset_full.csv')

d = df.dropna(axis='columns')
d.to_csv('/home/deepti/Downloads/Melbourne/Output/drop.csv') 


d=df.fillna(df.mean())
d.to_csv('/home/deepti/Downloads/Melbourne/Output/mmClean.csv')


d=df.interpolate(method ='linear',limit_direction ='both')
d.to_csv('/home/deepti/Downloads/Melbourne/Output/LiClean.csv')


from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
lb=LabelEncoder()
d["CouncilArea"] = lb_make.fit_transform(d["CouncilArea"])
d["Regionname"] = lb.fit_transform(d["Regionname"])
d.to_csv('/home/deepti/Downloads/Melbourne/Output/encoded.csv')


############################################################################################################
cor=d.corr()
cor['YearBuilt']

from scipy.stats import stats
tau, p_value = stats.kendalltau(pd.Series(d['Price']), pd.Series(d['Rooms']))
alpha = 0.05
if p_value > alpha:
	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p_value)
else:
	print('Samples are correlated (reject H0) p=%.3f' % p_value)
    
tau, p_value = stats.spearmanr(pd.Series(d['Price']), pd.Series(d['Rooms']))
alpha = 0.05
if p_value > alpha:
	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p_value)
else:
	print('Samples are correlated (reject H0) p=%.3f' % p_value)
    
    
    
lb_type=LabelEncoder()
d["Type"] = lb_type.fit_transform(d["Type"])
lb_method=LabelEncoder()
d["Method"] = lb_method.fit_transform(d["Method"])

lb_sub=LabelEncoder()
d["Suburb"] = lb_sub.fit_transform(d["Suburb"])

lb_sell=LabelEncoder()
d["SellerG"] = lb_sell.fit_transform(d["SellerG"])

d1=d

d1=d1.rename(index=str, columns={"Lattitude": "Latitude", "Longtitude": "Longitude"})
d1=d1.drop(['Address', 'Date', 'Latitude', 'Longitude'], axis='columns')



y=d1['Price']
X=d1.drop('Price', axis='columns')    

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)   #make predictions




import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
plt.scatter(
    d1['Rooms'],
    d1['Price'],
    c='black'
)
plt.scatter(
    d1['Rooms'],
    y_pred,
    c='blue',
    linewidth=2
)
plt.xlabel("No. of rooms")
plt.ylabel("Price of house")
plt.show()



import numpy as np
from sklearn import metrics
print "R2 Score: ", metrics.r2_score(y_test, y_pred)
print "MAE:", metrics.mean_absolute_error(y_test, y_pred)
print 'MSE:', metrics.mean_squared_error(y_test, y_pred)
print 'RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))



d1=d1[d1['Rooms'] < 9]



import xgboost
best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
best_xgb_model.fit(X_train,y_train)

y_pred = best_xgb_model.predict(X_test)

print "R2 Score: ", metrics.r2_score(y_test, y_pred)
print "MAE:", metrics.mean_absolute_error(y_test, y_pred)
print 'MSE:', metrics.mean_squared_error(y_test, y_pred)
print 'RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))


d1.boxplot(column="Price",by="Regionname")





