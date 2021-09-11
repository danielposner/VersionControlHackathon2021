import numpy as np
import scipy as sp
from scipy.stats import sem
import statistics as stat
import csv
import matplotlib.pyplot as plt
import matplotlib
from sklearn import*
from sklearn.linear_model import LinearRegression
from scipy.stats import iqr



## I'm implementing pandas for the first time here - so it may take a while
## Using numoy arrays and wines

with open("winequality-white-small.csv", 'r') as f:
    wines = list(csv.reader(f, delimiter=";"))
    ## Bug 2 -> Rounding to 2 decimal places in excel not python
    ## Bug 1 - for some reason - the column names are buggery
wines = np.array(wines[1:], float)
wines = wines.astype(int)

    ## Clean this up later - do ints for simplicity now
    ## wines[3] = returns a row
""" At this point -> all the data is in an array """
    ## wines[2,3] returns value 4th column third row [3]
    ## wines[:, 3] returns column at 3 [3] on end returns a value

rows = len(wines[3]) # returns total cols
cols = len(wines[:,2]) #returns total rows

def _mean():
    """Returns a single mean. Only to be called within a loop"""
    return stat.mean(wines[3])
    ### Convert to Loop also Median ###

def _median():
    """Returns a single median. Only to be called within a loop"""
    return stat.median(wines[3])

def _std():
    """ Returns standard deviation for a column"""
    return np.std(wines[:, 3])## 3 is a long term placeholder

def _standard_error():
    """standard deviation / sample size for a column"""
    return(sem(wines[:, 3]))

def _liner_regression_model():
    """Takes 2 values and spits out a LR
    Don't ask why it needs the newaxis"""
    model = LinearRegression(fit_intercept=True)
    model.fit(wines[3][:, np.newaxis], wines[5])
    xfit = np.linspace(0,10,1000) ## clean this up - lost - 391 PDSH Oreilly
    yfit= model.predict(xfit[:, np.newaxis])

def _inter_quartile_range():
    "Returns the IQR of an entire array (Q3 - Q1, 75% - 25%"
    return iqr(wines[:, 3])
    
    
    #print("Coefficient", model.coef_)
    #print("slope", model.coef_[0])
    #print("Intercept", model.intercept_)
    
    ##plt.scatter(wines[3], wines[5])
    ##plt.plot(xfit, yfit)
    ##plt.show()



