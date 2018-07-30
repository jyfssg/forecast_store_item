# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 22:23:03 2018

@author: Ethan Ji
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.stattools import adfuller

#read in .csv data file
data_train = pd.read_csv("train.csv", parse_dates = ["date"], index_col = "date")

#explore this data set
print (data_train.head())
print ("\n Data Types:")
print (data_train.dtypes)
data_train.index

m = max(data_train["store"])
n = max(data_train["item"])
k = data_train.shape[0]
p = int(k/(m*n))

#slice by different store and item
ts_train = data_train.iloc[0:p, 2]

#Determing rolling statistics
rolmean = pd.Series.rolling(ts_train, window=12).mean()
rolstd = pd.Series.rolling(ts_train, window=12).std()

#Plot rolling statistics:
orig = plt.plot(ts_train, color="blue",label="Original")
mean = plt.plot(rolmean, color="red", label="Rolling Mean")
std = plt.plot(rolstd, color="black", label = "Rolling Std")
plt.legend(loc="best")
plt.title("Rolling Mean & Standard Deviation")
plt.show(block=False)

#Perform Dickey-Fuller test:
print ("Results of Dickey-Fuller Test:")
dftest = adfuller(ts_train, autolag="AIC")
dfoutput = pd.Series(dftest[0:4], index=["Test Statistic","p-value","#Lags Used","Number of Observations Used"])
for key,value in dftest[4].items():
    dfoutput["Critical Value (%s)"%key] = value
print (dfoutput)

if dfoutput["p-value"] > 0.01:
      

    