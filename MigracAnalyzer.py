import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class UploadFromCSV:

    """#Преобразовывает Frame в два массива """
    @staticmethod    
    def ConvToArrays(my_dict): 
        keys = []
        values = []
        for k, v in my_dict.iterrows():
            keys.append(v['S'])
            values.append(v['N'])
        return (keys,values)

    def __init__(self, fileName):
        self.fileName = fileName
        self.datas = pd.DataFrame()

    def readFile(self, low_memory):
        try:
            self.datas = pd.read_csv(self.fileName,sep=';', low_memory=low_memory)
        except Exception as e:
            print(e)
        return self.datas

    def readFileSep(self, low_memory, sep):
        try:
            self.datas = pd.read_csv(self.fileName,sep=sep, low_memory=low_memory)
        except Exception as e:
            print(e)
        return self.datas

    def workAllPeopleFile(self):
        self.readFile(False)
        #for index, row in self.datas.iterrows():
        #    print(row['All2012'])
        return self.datas

    def workDirty(self, sep):
        self.readFileSep(False, sep)
        #for index, row in self.datas.iterrows():
        #    print(row)
        return self.datas

    def workLvlLife(self, sep):
        self.readFileSep(False, sep)
        for index, row in self.datas.iterrows():
            print(row)
        return self.datas

    def workUnemployed(self, sep):
        self.readFileSep(False, sep)
        for index, row in self.datas.iterrows():
            print(row)
        return self.datas

    def workRain(self, sep):
        self.readFileSep(False, sep)
        #for index, row in self.datas.iterrows():
        #    print(row)
        return self.datas

    def workCoords(self, sep):
        self.readFileSep(False, sep)
        #for index, row in self.datas.iterrows():
        #    print(row)
        return self.datas


 
