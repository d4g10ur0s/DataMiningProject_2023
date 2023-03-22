
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import math

import os

def datasetToFolds(data):
    # shuffle data
    data = data.sample(n=len(data),axis=0,ignore_index=True)
    #splice data to 5-fold
    fnum = int(input("Number of Folds : "))
    ffold = []
    batchlen = math.floor(len(data)/fnum)
    rem = len(data) - batchlen * fnum
    for i in range(fnum):
        if(i==fnum-1):
            ffold.append(data.iloc[i*batchlen:(i+1)*batchlen+rem][:].reset_index())
        else:
            ffold.append(data.iloc[i*batchlen:(i+1)*batchlen][:].reset_index())
    return ffold

def data_reader():
    # read csv
    dpath = os.getcwd()
    data = pd.read_csv(dpath+"\\data.csv")
    return data

def main():
    #1. read csv
    data = data_reader().fillna(0)
    #2. find countries
    countries = []
    for i in data.iloc[:]["Entity"]:
        if i not in countries :
            countries.append(i)
    #3. one dataframe for each country
    df_dict = {}
    for i in countries :
        df_dict[i] = data.loc[:][data["Entity"]==i].reset_index().drop(columns="index")
    #4. plot population per day for each
    for i in countries :
        proc = df_dict[i]
        dd1 = proc.loc[1:][["Deaths","Cases"]].reset_index().drop(columns="index")#daily deaths
        dd2 = proc.loc[:len(proc.index)-1][["Deaths","Cases"]].reset_index().drop(columns="index")#daily deaths
        frames = [pd.DataFrame(data=[proc.loc[0]]), dd1-dd2]

        dailyDeaths = pd.concat(frames, join="inner").reset_index().drop(columns="index").loc[:len(proc.index)-1]["Deaths"]

        dailyCases = pd.concat(frames, join="inner").reset_index().drop(columns="index").loc[:len(proc.index)-1]["Cases"]
        dailyCases.loc[:].replace(to_replace=0.0,value=1,inplace=True)#because some days cases were 0...

        plt.plot(range(len(proc.loc[:]["Date"])),dailyDeaths/dailyCases)
        plt.title(i)
        plt.savefig("Graphs\\"+i+"_DailyDeaths_Div_DailyCases"+".png")
        plt.clf()
        '''
        plt.show()
        '''


if __name__ == "__main__":
    main()
