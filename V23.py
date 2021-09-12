import numpy as np
from numpy import cov
import scipy
import scipy.stats as st
from scipy import stats
from scipy.stats import sem
import statistics as stat
import csv
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import pandas as pd
from sklearn import*
from sklearn.linear_model import LinearRegression
from scipy.stats import iqr
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import itertools
from scipy.stats import linregress
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import bartlett


                ### 4 tabbed comment means it's informative



## Implement pandas if you have the time at the end
## Using numpy arrays and wines

with open("winequality-white-small.csv", 'r') as f:
    wines = list(csv.reader(f, delimiter=";"))
    ## Bug 1 - for some reason - the column names are buggery
names = wines[0]
names = names[0].split(";")
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

## Maybe if nothing else works use below
## https://medium.com/the-code-monster/split-a-dataset-into-train-and-test-datasets-using-sk-learn-acc7fd1802e0


##############################################################################################################################################
################## DEPENDENCY KNOWLEDGE FUNCTIONALITY ######################## ################# DEPENDENCY KNOWLEDGE FUNCTIONALITY ##########
##############################################################################################################################################

## Do this when I start later modelling -> Same goes for categorical/Ordinal
    
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
        Returns the correlation co-efficient and the P value""" #bar 95% and .3
        pearson = pearsonr(a, b)

        if pearson[0] > 0.3 or pearson[0] < -.3: ## Using .3 as the bar for pearson coefficient - medium relation
            if pearson[1] < 0.05:
                self.tag.append("1")
                self.tag.append("1")

        elif pearson[1] < 0.05:
            self.tag.append("0")
            self.tag.append("1")
        
        else:
            self.tag.append("0")
            self.tag.append("0")


            
        return [pearson[0], pearson[1]]
    
        #Pearson’s correlation coefficient.
        #Pearson 2 tailed P value


####################################################################################################
    ## Not using spearmans yet - later implementation for non-continuous variables

    def _spearman_correlation_test(self, a, b):
        """Does the spearman correlation for two variables
        @return the P value and correlation value
        Typically used for ordinal variables
        """
        x = spearmanr(a,b)
        return [x[0], x[1]]
        #Returns Correlation
        ## Returns P value against null


    #def _cramers_v_contingency_test(self,a,b):
    # DO not use - only accepts integer arrays
    
    def _equal_pop_variance_bartlett(self, a, b):
        stat, p = bartlett(a, b)
        if p > 0.05:
            self.tag.append("1")
        else:
            self.tag.append("0")
        return [p] #could return stat - pass

    def _v1_normal_dist_test(self,a,b):
        alpha = 1e-3
        k2, p = stats.normaltest(a)
        if p < alpha:
            self.tag.append("1")
        else:
            self.tag.append("0")
        return [p]
        
        
        
    

####################################################################################################

    def _dependency_regression_1(self, a,b):
    ## https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#examples
    ## -using-sklearn-feature-selection-mutual-info-regression
        """Measures dependency between variables
        """
        #### ERROR -> Not sure how to read the data
        x = sklearn.feature_selection.mutual_info_regression(a.reshape(-1, 1), b)
        y = sklearn.feature_selection.mutual_info_regression(b.reshape(-1, 1), a)
        if x > 0.75 and y > 0.75: ## Semiarbitrary - attach email
            self.tag.append("1")
            self.tag.append("1")

        elif x > 0.75:
            self.tag.append("1")
            self.tag.append("0")

        elif y > 0.75:
            self.tag.append("0")
            self.tag.append("1")

        else:
            self.tag.append("0")
            self.tag.append("0")
        
        return [x, y]


    def _liner_regression_model(self, a,b):

        ### KEY - use a training set https://medium.com/the-code-monster/split-a-dataset-into-train-and-test-datasets-using-sk-learn-acc7fd1802e0
        #### Depends on which variavle is dependnet -> clean up later
        """Takes 2 values and spits out a LR
        Incredibly basic - possibly build on this later
        Wants 2 dependent on 1 but so be it
        """
        x = linregress(a, b)
        #y = linregress(b,a)
        #self.tag.append("0")
        #self.tag.append("0")
        if x[2] > .3: #corr coeff medium
            self.tag.append("1")
        else:
            self.tag.append("0")
        if x[3] < 0.05:
            self.tag.append("1")
        else:
            self.tag.append("0")
            
            

        #return x[0], x[1] Slope and intercept - useless here
        return  x[2], x[3] ## Slope, intercept, Correlation Coefficient, P-value (wald)

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
        ### CHECK THIS
        return [st.t.interval(alpha=0.95, df=len(a)-1, loc=np.mean(a), scale=st.sem(a))]
        ### CHECK THIS

    def _95_confidence_interval_true_mean_over_30_samples(self, a, b):
        """ 95% Confidence interval for true mean
        This formula is preferred when we have over 30 samples and can assume normal distrubution)
        Gets very repetetive (deal with it)"""
        
        ### CHECK THIS
        return [st.norm.interval(alpha=0.95, loc=np.mean(a), scale=st.sem(a))]
        ### CHECK THIS


    
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

        self.full_list = [
                          self._two_tailed_t_test, self._pearson_correlation_test,
                          self._dependency_regression_1, self._liner_regression_model,
                          self._equal_pop_variance_bartlett, self._v1_normal_dist_test
                          ]

## Not testing ordinal variables here
## Removed useless data
                            ## self._mean, self._median, self._std, self._standard_error, self._inter_quartile_range,
                          #self._listing_Confidence_Intervals, self._spearman_correlation_test,


        special = ["Identified Variables"]
        for n in self.full_list: #### RECHECK TAGS FOR ALL SETS
            if n == self._mean:
                special.append("Mean")
            if n == self._median:
                special.append("Median")
            if n == self._two_tailed_t_test:
                special.append("T Statistic")
                special.append("2TTTest PValue") #large means similar - small means different
            if n == self._std:
                special.append("STD Dev")
            if n == self._standard_error:
                special.append("STD Error")
            if n == self._inter_quartile_range:
                special.append("IQR")
            if n == self._listing_Confidence_Intervals:
                special.append("95% CI")
            if n == self._pearson_correlation_test:
                special.append("Pearson Correlation Coefficient") ## No data screening - outliers have large effect
                special.append("Pearson 2T PValue") 
            if n == self._spearman_correlation_test:
                special.append("Spearman0") ## Add tag ## Bad naming
                special.append("Spearman1") ## Add tag
            if n == self._dependency_regression_1:
                special.append("Dregression AB") ## Add tag ## FIX BUG -> STILL LIST ## Bad naming
                special.append("Dregression BA") ## Add tag ## FIX BUG
            if n == self._liner_regression_model:
                #special.append("LReg Slope") ## Bad naming
                #special.append("LReg Intercept") Worthless for this purpose
                special.append("LReg Corr Coeff")
                special.append("LReg Pval (Wald)")
            if n == self._equal_pop_variance_bartlett:
                special.append("wantHIGH Pval EqlPopVar Bartlett")
            if n == self._v1_normal_dist_test:
                special.append("Var1 NormalDist want < 1e-3")


                ### Change return and data - 1,2,3,4,5 AB, 1,2,3,4,5 ??

                ##Slope, intercept, Correlation Coefficient, P-value (wald)
            
            
        special.append("Tag")
        self.special = special
        #print(self.special)

            



    def iterative_cycle(self):
        allcols = list(itertools.combinations(range(len(rangerows)), 2))
        with open('HackathonOutput.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.special)
            for pair in allcols:
                self.tag = []
                results = [names[pair[0]] + names[pair[1]]]
                for function in self.full_list:
                    x = function(wines[:,pair[0]], wines[:,pair[1]])
                    for n in x:
                        results.append(n)
                        
                results.append(self.tag)
                ## Adds tag to end -> Mess with joining at end during cleanup
                writer.writerow(results) ## Writes in

    def cleaner1(self):
        """This got clunky but eh
        Essentially I'm removing the tag, and turning all irrelevant
        data to zeros, then saving that to a new file"""
        
        with open("HackathonOutput.csv", 'r') as f:
            d1 = list(csv.reader(f))
            headers = d1[0]
            
            x = ''.join(c for c in d1[-2][-1] if c.isdigit()) 
            count = 1
            while count < len(d1):
                x = ''.join(c for c in d1[count][-1] if c.isdigit())
                d1[count][-1] = list(x)
                count+= 1 ## This turns the tag into a list of digits
##
            count = 1
            while count < len(d1):
                x = 1
                for n in d1[count][-1]:
                    if n == '0':

                        d1[count][x] = 0
                    x += 1
                count += 1

            with open('HackathonCleaned1.csv', 'w', newline='') as file2:
                writer = csv.writer(file2)
                for n in range(len(d1)):
                    writer.writerow(d1[n][:-1])
                    ## This got clunky but eh
                    ## Essentially I'm removing the tag, and turning all irrelevant
                    ## data to zeros
                    

    
                    
                
                
##                    
                    


            
            
            


            #DOn't do array or pandas work - keep it in lists for now
            #d2 = np.array(wines[1:], float) ## WOrking to here
            print(d1[2])
            #print(d1)
            #print(d1[2, -1]) #returns column at 3 [3] on end returns a value
#


            


            

        


                 



             

             

            
             
        # Check the panda for all files with a live entry
        # Wipe out all the completely dead columns from the panda
        # Write the panda to excel
        # Store a list of all live variable numbers
        

##with open("winequality-white-small.csv", 'r') as f:
##    wines = list(csv.reader(f, delimiter=";"))
##    ## Bug 1 - for some reason - the column names are buggery
##names = wines[0]
##names = names[0].split(";")
##wines = np.array(wines[1:], float)



        
    ## Def iterative cycle 2
        ## New Tests - copy some, write some
        ## All live variables from Cleaning_Cy
        ## Run all live variables against remaining singles (not each other)
        ## Test for overlap
        ## Returns another bulky output with tags

        
        #with open('HackathonOutput2.csv', 'w', newline='') as file:
            ## Split tag back into identifying each row
            ## 
        
                    
                        ## this does returns - set tag here too
                    
                
            

kk = Main()
kk.function_list()
kk.iterative_cycle()
kk.cleaner1()
