import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time

class Handeler():
    def __init__(self):
        super(Handeler, self).__init__()
        self.data = None

    #Set the dataframe
    def inputs(self, path):
        self.data = pd.read_csv(path)
        #List of varibales in my dataset
        return self.data

    #get columns name
    def _variabls(self):
        return self.data.columns.values.tolist()

    def _null(self):
        return self.data.isnull().sum()

    def _drop(self):
        l_drop = self.data.columns[self.data.isnull().any()].tolist()
        for x in l_drop:
            self.data = self.data.drop([x], axis=1)
        self.data = self.data.drop(["Id"], axis=1)
        return self.data

#This section is for test and by default it load data
if __name__ == '__main__':
    file = Handeler()
    path = os.path.abspath(os.path.join(Path().absolute(), os.pardir)) + "/data/train.csv"
    out = file.inputs(path)

    #Test du commit pycharm
    out = file._variabls()
    print("the number of variables that we have is : \n", out)
    #print(file._null() != 0)

    out = file._drop()
    #print("the number of variables that we have is : \n", out)



