#V10.2 XLS Writer (tagging)




from openpyxl import Workbook
from openpyxl.styles import colors
from openpyxl.styles import Font, Color
import itertools


item1 = ["cat", "Dog", "Fish"]
ft = Font(color="FF0000")
for n in item1:
    n.font = ft
item2 = ["Bat", "Rat", "Gnat"]

items = item1+item2
items =list(itertools.chain(item1, item2))
    ## This should be a name tag attached to the function name
    ## List length = number of returned outputs
item1returns = [1,2,3]
item2returns = [4,5,6]
itemss =list(itertools.chain(item1returns, item2returns))


wb = Workbook()
ws =  wb.active
ws.title = "HackathonOutput"
ws.append(items)
ws.append(itemss)


wb.save(filename = 'HackathonOutput.xlsx')









##import csv
##import itertools
##
##with open('HackathonOutput.csv', 'w', newline='') as file:
##    #### Writes data to tuple
##    #### make a list of titles per header
##
##    item1 = ["cat", "Dog", "Fish"]
##    item2 = ["Bat", "Rat", "Gnat"]
##
##    items = item1+item2
##    items =list(itertools.chain(item1, item2))
##        ## This should be a name tag attached to the function name
##        ## List length = number of returned outputs
##    item1returns = ["1", "2", "3"]
##    item2returns = ["4", "5", "6"]
##    itemss =list(itertools.chain(item1returns, item2returns))
##        ## Function return
##
##    
##    writer = csv.writer(file)
##    writer.writerow(items)
##    writer.writerow(itemss)
##        #### Put the list of all return values here
##
