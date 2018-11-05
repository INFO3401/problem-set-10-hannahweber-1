# Monday/Wednesday Problems - worked with Taylor, Marissa, Jacob, Jack
import csv
import pandas as pd

# Part A
class AnalysisData:

    def __init__(self):
        self.dataset = []
        self.variables = []

    def parseFile(self, filename):
        
        #read csv into a panda
        self.dataset = pd.read_csv(filename)
        
        # set all of the column names, except the first one, into the variables variable
        for column in self.dataset.columns.values:
            if column != "competitorname":
                self.variables.append(column)
        

# Part B

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class LinearAnalysis:
    
    def __init__(self, targetY_input):
        self.bestX = ""
        self.targetY = targetY_input
        self.fit = ""
        
    def runSimpleAnalysis(self, data):
        best_rscore = -1
        best_variable = ""
        
        for column in data.variables:
                if column != self.targetY:
                        independent_variable = data.dataset[column].values
                        # this line fixed an error that was occurring when running the line above
                        independent_variable = independent_variable.reshape(len(independent_variable), 1)
                        
        regr = LinearRegression()
        regr.fit(independent_variable, data.dataset[self.targetY])
        #regr.fit(<candy>, <sugar>)
        prediction = regr.predict(independent_variable)
        #regr.predict(<candy>)
        r_score = r2_score(data.dataset[self.targetY], prediction)
        #r2_score(<sugar>, <predicted values>)
        
        if r_score > best_rscore:
            best_rscore = r_score
            best_variable = column
            
        self.bestX = best_variable
        print(best_variable, best_rscore)
        
        
# Part C

from sklearn.linear_model import LogisticRegression

class LogisticAnalysis:
    
    def __init__(self, targetY_input):
        self.bestX = ""
        self.targetY = targetY_input
        self.fit = ""
        
    def runSimpleAnalysis(self, data):
        best_rscore = -1
        best_variable = ""
        
        for column in data.variables:
                if column != self.targetY:
                        independent_variable = data.dataset[column].values
                        # this line fixed an error that was occurring when running the line above
                        independent_variable = independent_variable.reshape(len(independent_variable), 1)
                        
        regr = LogisticRegression()
        regr.fit(independent_variable, data.dataset[self.targetY])
        #regr.fit(<candy>, <sugar>)
        prediction = regr.predict(independent_variable)
        #regr.predict(<candy>)
        r_score = r2_score(data.dataset[self.targetY], prediction)
        #r2_score(<sugar>, <predicted values>)
        
        if r_score > best_rscore:
            best_rscore = r_score
            best_variable = column
            
        self.bestX = best_variable
        print(best_variable, best_rscore)
        
# Last Week
analysisData = AnalysisData()
analysisData.parseFile("candy-data.csv")

# Last Week
linearAnalysis = LinearAnalysis("sugarpercent")
linearAnalysis.runSimpleAnalysis(analysisData)

# Problem 1
logisticAnalysis = LogisticAnalysis("chocolate")
logisticAnalysis.runSimpleAnalysis(analysisData)

# They both find the same optimal variable "winpercent" but linear regression fits the data better.