import numpy as np
import scipy as sp
import csv


## I'm implementing pandas for the first time here - so it may take a while
## Using numoy arrays and wines

class Constructor:
    """ We analyze data in columns. Tuples are the individual data unit """

    def __init__(self):
        self.wines = create_dataset()
    
    def create_dataset(self):

        with open("winequality-white-small.csv", 'r') as f:
            wines = list(csv.reader(f, delimiter=";"))
            ## Bug 2 -> Rounding to 2 decimal places in excel not python
            ## Bug 1 - for some reason - the column names are buggery
        import numpy as np
        wines = np.array(wines[1:], float)
        wines = wines.astype(int)
            ## Clean this up later - do ints for simplicity now
            ## third_wine = wines[3, :]
            ## third_wine[2] returns the 3rd wine
            ## wines[3] = returns a row
        """ At this point -> all the data is in an array """
            ## wines[2,3] returns value 4th column third row
            ## wines[:, 3] returns column at 3
        print("cat")
        return wines

    #def number_of_cols_and_rows(self):
        #print(self.wines[1].size())

    #def standard_deviation():
        #pass


base = Constructor()
base.create_dataset()
#base.number_of_cols_and_rows()


    #number_of_cols_and_rows(base)


    





