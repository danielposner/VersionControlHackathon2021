import numpy as np
from numpy import cov
import scipy
import scipy.stats as st
from scipy.stats import sem
import statistics as stat
import csv
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn import*
from sklearn.linear_model import LinearRegression
from scipy.stats import iqr
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import itertools
from scipy.stats import linregress
from sklearn.feature_selection import mutual_info_regression

                ### 4 tabbed comment means it's informative



## I'm implementing pandas for the first time here - so it may take a while
## Using numpy arrays and wines

with open("winequality-white-small.csv", 'r') as f:
    wines = list(csv.reader(f, delimiter=";"))
    ## Bug 1 - for some reason - the column names are buggery
names = wines[0]
names[0].split(";")
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



## ? https://scikit-learn.org/stable/modules/partial_dependence.html

##
# DEPENDENCY TESTING (train/test dataset?)
##



## https://medium.com/the-code-monster/split-a-dataset-into-train-and-test-datasets-using-sk-learn-acc7fd1802e0


##############################################################################################################################################
################## DEPENDENCY KNOWLEDGE FUNCTIONALITY ######################## ################# DEPENDENCY KNOWLEDGE FUNCTIONALITY ##########
##############################################################################################################################################
    
## One way ANOVA
## Paired T test
## One way Repeated ANOVA
## Check 2 way ANOVA

    
##############################################################################################################################################
################## MAIN FUNCTIONALITY ######################## MAIN FUNCTIONALITY ########################MAIN FUNCTIONALITY #################
##############################################################################################################################################





class Main:
    def __init__(self):
        self.tag = []

    def _pearson_correlation_test(self, a, b):
        """Tests two columns for covariance
        Returns the correlation co-efficient"""
        pearson = pearsonr(a, b)
        return [pearson[0], pearson[1]]

    def _spearman_correlation_test(self, a, b):
        """Does the spearman correlation for two variables
        @return the P value and correlation value
        Typically used for ordinal variables
        """
        x = spearmanr(a,b)
        return [x[0], x[1]]

    def _dependency_regression_1(self, a,b):
    ## https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#examples
    ## -using-sklearn-feature-selection-mutual-info-regression
        """Measures dependency between variables"""
        #### ERROR -> Not sure how to read the data
        x = sklearn.feature_selection.mutual_info_regression(a.reshape(-1, 1), b)
        y = sklearn.feature_selection.mutual_info_regression(b.reshape(-1, 1), a)
        return [x, y]

    def _liner_regression_model(self, a,b):

        ### KEY - use a training set https://medium.com/the-code-monster/split-a-dataset-into-train-and-test-datasets-using-sk-learn-acc7fd1802e0
        #### Depends on which variavle is dependnet -> clean up later
        """Takes 2 values and spits out a LR
        Incredibly basic - possibly build on this later
        """
        x = linregress(a, b)
        y = linregress(b,a)
        return [x,y]

#########################################
#############UNBUILT#####################
#########################################
    def _dependent_t_test(a,b):
        """Applies for dependent variables
        DO DEPENDENCE TEST FIRST"""

    ## Find another test or two

#########################################
#############UNBUILT#####################
#########################################


    def _std(self, a, b):
        """ Returns standard deviation for a column
        B is a dead variable, needs to be imputted due to other loops"""
        self.tag.append("0")
        return [np.std(a)]


    def _95_confidence_interval_true_mean_under_30_samples(a, b):
        """ 95% Confidence interval for true mean
        This formula is preferred when we have less than 30 samples (we can't assume mean is normally distrubuted)"""
        ### Adapt to other things
        return [st.t.interval(alpha=0.95, df=len(a)-1, loc=np.mean(a), scale=st.sem(a))]

        #not worth implenting unless I get time
        #print("95%", st.t.interval(alpha=0.95, df=len(b)-1, loc=np.mean(b), scale=st.sem(b)))


    def _95_confidence_interval_true_mean_over_30_samples(self, a, b):
        """ 95% Confidence interval for true mean
        This formula is preferred when we have over 30 samples and can assume normal distrubution)
        Gets very repetetive (deal with it)"""
        
        ### Adapt to other things
        return [st.norm.interval(alpha=0.95, loc=np.mean(a), scale=st.sem(a))]
        #print(st.norm.interval(alpha=0.95, loc=np.mean(b), scale=st.sem(b)))

        ## TAG THE LOt


    
    def _standard_error(self, a,b):
        """standard deviation / sample size for a column
        B is a dead variable, needs to be imputted due to other loops"""
        self.tag.append("0")
        return [sem(a)]

    def _inter_quartile_range(self, a,b):
        """Returns the IQR of an entire array (Q3 - Q1, 75% - 25%
        B is a dead variable, needs to be imputted due to other loops"""
        self.tag.append("0")
        return [iqr(a)]

    def _listing_Confidence_Intervals(self, a, b):
        self.tag.append("0")
        if len(rangecols) > 30:
            return [self._95_confidence_interval_true_mean_over_30_samples(a, b)]
        else:
            return [self._95_confidence_interval_true_mean_under_30_samples(a, b)]

#######################
    def _mean(self, a, b):
        """Returns a single mean. Only to be called within a loop
        B is a dead variable, needs to be imputted due to other loops"""
        self.tag.append("0")
        return([stat.mean(a)])

    def _median(self, a, b):
        """Returns a single median. Only to be called within a loop
        B is a dead variable, needs to be imputted due to other loops"""
        self.tag.append("0")
        return([stat.median(a)])

    def _two_tailed_t_test(self, a,b):
        """ Assumption of normality and equal variance """
        ### Later I will want to create tests for the other T-Tests
        x = (scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate',
                              permutations=None, random_state=None, alternative='two-sided', trim=0))

        if x[1] < 0.05: #If p value is 5% or less
            self.tag.append("1")
            self.tag.append("1")
        else:
            self.tag.append("0")
            self.tag.append("0")
            
        return [x[0], x[1]]

                                                        ### All data is contained in https://docs.scipy.org/
                                                        ### doc/scipy/reference/generated/scipy.stats.ttest_ind.html

    

    def function_list(self):
        """All implemented functions are listed here. If time allows, allow editing of this list in GUI
        Pearson Correlation DONE"""
        #Test Case
        #dt1 may be worthless
        self.full_list = [self._mean, self._median, self._two_tailed_t_test, self._std, self._standard_error,
                          self._inter_quartile_range, self._listing_Confidence_Intervals, self._pearson_correlation_test,
                          self._spearman_correlation_test, self._dependency_regression_1, self._liner_regression_model]

        special = ["Identified Variables"]
        for n in self.full_list:
            if n == self._mean:
                special.append("Mean")
            if n == self._median:
                special.append("Median")
            if n == self._two_tailed_t_test:
                special.append("CHECK 2TTTest")
                special.append("2TTTest PValue")
            if n == self._std:
                special.append("STD Dev")
            if n == self._standard_error:
                special.append("STD Error")
            if n == self._inter_quartile_range:
                special.append("IQR")
            if n == self._listing_Confidence_Intervals:
                special.append("95% CI")
            if n == self._pearson_correlation_test:
                special.append("Pearson0") ## Bad naming
                special.append("Pearson1") 
            if n == self._spearman_correlation_test:
                special.append("Spearman0") ## Add tag ## Bad naming
                special.append("Spearman1") ## Add tag
            if n == self._dependency_regression_1:
                special.append("DepReg data1") ## Add tag ## FIX BUG -> STILL LIST ## Bad naming
                special.append("DepReg data2") ## Add tag ## FIX BUG
            if n == self._liner_regression_model:
                special.append("LRM Data1") ## Bad naming
                special.append("LRM Data2") ### Change return and data - 1,2,3,4,5 AB, 1,2,3,4,5 ??
            

            #self._std, self._standard_error,
                          #self._inter_quartile_range, self._listing_Confidence_Intervals
            
        special.append("Tag")

        self.special = special
            

        print(self.full_list)


        ## Final List
        #self._listing_Confidence_Intervals()
        #self.full_list = [_pearson_correlation_test, _spearman_correlation_test, _mean, _median,
                        ## _std, _standard_error, _inter_quartile_range, _liner_regression_model,
                        ## _two_tailed_t_test, dt1


            



    def iterative_cycle(self):
        allcols = list(itertools.combinations(range(len(rangerows)), 2))
        with open('HackathonOutput.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.special)
            for pair in allcols:
                self.tag = []
                results = [names[pair[0]], [pair[1]]]
                for function in self.full_list:
                    x = function(wines[:,pair[0]], wines[:,pair[1]])
                    for n in x:
                        results.append(n)
                    #results.append(x)
                        
                    #writer.writerow(items)
                results.append(self.tag)
                ## Adds tag to end -> Mess with joining at end during cleanup
                writer.writerow(results) ## Writes in

    #def cleaning_cycle_1(self):
        #with open('HackathonOutput2.csv', 'w', newline='') as file:
        
                    
                        ## this does returns - set tag here too
                    
                
            

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
kk.cleaning_cycle_1()
