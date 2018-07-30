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

#############################################################################
#Build a sef defined function to check time series' stationary
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color="blue",label="Original")
    mean = plt.plot(rolmean, color="red", label="Rolling Mean")
    std = plt.plot(rolstd, color="black", label = "Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard Deviation")
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic","p-value","#Lags Used","Number of Observations Used"])
    for key,value in dftest[4].items():
        dfoutput["Critical Value (%s)"%key] = value
    print (dfoutput)
#End of the self defined function
#############################################################################

data_train = pd.read_csv("train.csv", parse_dates = ["date"], index_col = "date")

print (data_train.head())
print ("\n Data Types:")
print (data_train.dtypes)
data_train.index

m = max(data_train["store"])
n = max(data_train["item"])
k = data_train.shape[0]
l = int(k/(m*n))

ts_train = data_train.iloc[0:l, 2]

plt.plot(ts_train)

test_stationarity(ts_train)
    