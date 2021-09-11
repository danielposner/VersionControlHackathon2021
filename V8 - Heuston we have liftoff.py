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
import itertools
from scipy.stats import linregress


                ### 4 tabbed comment means it's informative



## I'm implementing pandas for the first time here - so it may take a while
## Using numpy arrays and wines

with open("winequality-white-small.csv", 'r') as f:
    wines = list(csv.reader(f, delimiter=";"))
    ## Bug 1 - for some reason - the column names are buggery
wines = np.array(wines[1:], float)

                ## wines[3] = returns a row
                ##At this point -> all the data is in an array """
                ## wines[2,3] returns value 4th column third row [3]
                ## wines[:, 3] returns column at 3 [3] on end returns a value

rows = len(wines[3]) 
cols = len(wines[:,2]) 
rangerows = range(rows)
rangecols = range(cols)
    ## Bug 2 - Rows mean cols and cols mean rows. No functional difference just makes code ugly


################################################################
######## FINISHED FUNCTIONS ######## FINISHED FUNCTIONS ########
################################################################

def _mean(a, b):
    """Returns a single mean. Only to be called within a loop
        B is a dead variable, needs to be imputted due to other loops"""
    print(stat.mean(a))

def _median(a, b):
    """Returns a single median. Only to be called within a loop
        B is a dead variable, needs to be imputted due to other loops"""
    print(stat.median(a))

def _std(a, b):
    """ Returns standard deviation for a column
    B is a dead variable, needs to be imputted due to other loops"""
    print(np.std(a))

def _standard_error(a,b):
    """standard deviation / sample size for a column
B is a dead variable, needs to be imputted due to other loops"""
    print(sem(a))

def _inter_quartile_range(a,b):
    """Returns the IQR of an entire array (Q3 - Q1, 75% - 25%
    B is a dead variable, needs to be imputted due to other loops"""
    print(iqr(a))

def _95_confidence_interval_true_mean_under_30_samples(a, b):
    """ 95% Confidence interval for true mean
    This formula is preferred when we have less than 30 samples (we can't assume mean is normally distrubuted)"""
    ### Adapt to other things
    print("95%", st.t.interval(alpha=0.95, df=len(a)-1, loc=np.mean(a), scale=st.sem(a)))
    print("95%", st.t.interval(alpha=0.95, df=len(b)-1, loc=np.mean(b), scale=st.sem(b)))


def _95_confidence_interval_true_mean_over_30_samples(a, b):
    """ 95% Confidence interval for true mean
    This formula is preferred when we have over 30 samples and can assume normal distrubution)
    Gets very repetetive (deal with it)"""
    ### Adapt to other things
    print(st.norm.interval(alpha=0.95, loc=np.mean(a), scale=st.sem(a)))
    print(st.norm.interval(alpha=0.95, loc=np.mean(b), scale=st.sem(b)))


def _pearson_correlation_test(a, b):
    """Tests two columns for covariance
    Returns the correlation co-efficient"""
    pearson = pearsonr(a, b)
    print(pearson[0])
    return(pearson[0])

def _spearman_correlation_test(a, b):
    """Does the spearman correlation for two variables
    @return the P value and correlation value
    Typically used for ordinal variables
    """
    x = spearmanr(a,b)
    print("Correlation", x[0])
    print("P value", x[1].round(2)) ## Round everything to two digits

def _liner_regression_model(a,b):
    #### Depends on which variavle is dependnet -> clean up later
    """Takes 2 values and spits out a LR
    Incredibly basic - possibly build on this later
    """
    x = linregress(a, b)
    y = linregress(b,a)
    print(x, y)

def _dependent_t_test(a,b):
    """Applies for dependent variables
    DO DEPENDENCE TEST FIRST"""


def _two_tailed_t_test(a,b):
    """ Assumption of normality and equal variance """
    ### Later I will want to create tests for the other T-Tests
    print( scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate',
                          permutations=None, random_state=None, alternative='two-sided', trim=0))
                                                        ### All data is contained in https://docs.scipy.org/
                                                        ### doc/scipy/reference/generated/scipy.stats.ttest_ind.html
################################################################
######## FINISHED FUNCTIONS ######## FINISHED FUNCTIONS ########
################################################################






    
##############################################################################################################################################
################## MAIN FUNCTIONALITY ######################## MAIN FUNCTIONALITY ########################MAIN FUNCTIONALITY #################
##############################################################################################################################################

class Main:
    def __init__(self):
        pass

    def _listing_Confidence_Intervals(self):
        if len(rangecols) > 30:
            self.full_list.append(_95_confidence_interval_true_mean_over_30_samples)
        else:
            self.full_list.append(_95_confidence_interval_true_mean_under_30_samples)

    

    def function_list(self):
        """All implemented functions are listed here. If time allows, allow editing of this list in GUI
        Pearson Correlation DONE"""
        #Test Case
        self.full_list = [_liner_regression_model]

        print(self.full_list)


        ## Final List
        #self._listing_Confidence_Intervals()
        #self.full_list = [_pearson_correlation_test, _spearman_correlation_test, _mean, _median,
                        ## _std, _standard_error, _inter_quartile_range, _liner_regression_model,
                        ## _two_tailed_t_test


            



    def iterative_cycle(self):
        allcols = list(itertools.combinations(range(len(rangerows)), 2))
        for pair in allcols:
            print(pair)
            for function in self.full_list:
                function(wines[:,pair[0]], wines[:,pair[1]])
                
            

##
##
##        ## First set up functionality here -> then focus on optimizing using NUMPY kit
##        index = 0
##        while index < len(rangerows) -1:
##            print(index, index+1)
##            _pearson_correlation_test(wines[:,index], wines[:,index+1])
##            index += 1

            ##### RECURSION
        
            #for row in rangerows:
#print(rangerows)               
kk = Main()
kk.function_list()
kk.iterative_cycle()
