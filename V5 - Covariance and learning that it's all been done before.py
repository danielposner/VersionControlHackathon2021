import numpy as np
from numpy import cov
import scipy
import scipy.stats as st
from scipy.stats import sem
import statistics as stat
import csv
import matplotlib.pyplot as plt
import matplotlib
from sklearn import*
from sklearn.linear_model import LinearRegression
from scipy.stats import iqr
from scipy.stats import pearsonr
from scipy.stats import spearmanr





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
rangerows = range(rows)
rangecols = range(cols)

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

def _two_tailed_t_test():
    """ Assumption of normality and equal variance """
    ### Later I will want to create tests for the other T-Tests
    return scipy.stats.ttest_ind(wines[:,3], wines[:,5], axis=0, equal_var=True, nan_policy='propagate',
                          permutations=None, random_state=None, alternative='two-sided', trim=0)
    ### All data is contained in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

def _95_confidence_interval_true_mean_under_30_samples():
    """ 95% Confidence interval for true mean
    This formula is preferred when we have less than 30 samples (we can't assume mean is normally distrubuted)"""
    ### Adapt to other things
    return st.t.interval(alpha=0.95, df=len(wines[:,3])-1, loc=np.mean(wines[:,3]), scale=st.sem(wines[:,3]))

def _95_confidence_interval_true_mean_over_30_samples():
    """ 95% Confidence interval for true mean
    This formula is preferred when we have over 30 samples and can assume normal distrubution)"""
    ### Adapt to other things
    return st.norm.interval(alpha=0.95, loc=np.mean(wines[:,3]), scale=st.sem(wines[:,3]))


def _pearson_correlation_test():
    """Tests two columns for covariance
    Returns the correlation co-efficient"""
    pearson = pearsonr(wines[:,3], wines[:,5])
    print(pearson[0])
    return(pearson[0])

def _spearman_correlation_test():
    """Does the spearman correlation for two variables
    @return the P value and correlation value
    Typically used for ordinal variables
    """
    x = spearmanr(wines[:,3], wines[:,5])
    print("Correlation", x[0])
    print("P value", x[1].round(2)) ## Round everything to two digits
    
##############################################################################################################################################
################## MAIN FUNCTIONALITY ######################## MAIN FUNCTIONALITY ########################MAIN FUNCTIONALITY #################
##############################################################################################################################################

class main:
    def __init__():
        pass

    def iterative_cycle(): ## First set up functionality here -> then focus on optimizing using NUMPY kit
        for column in rangecols:
            for row in rangerows:
                
