# Graphs
import seaborn as sn
import matplotlib.pyplot as plt
# Processing
import pandas as pd
import numpy as np

import math

import os

from makeGraphs import graphByContinent

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
    #2. find continents and countries
    continents = data["Continent"].unique().tolist()
    countries = data["Entity"].unique().tolist()
    #3. one dataframe for each country ,one entry for each continent
    graphByContinent(continents,data)
    '''
    df_dict = {}
    for i in continents :
        df_dict[i] = []
    for i in countries :
        temp = data.loc[:][data["Entity"]==i].reset_index().drop(columns="index")
        df_dict[temp.loc[0]["Continent"]].append(temp)
    '''
    '''
    #4. plot population per day for each
    for i in continents :
        process = df_dict[i]
        total = None
        for proc in process :
            dd1 = proc.loc[1:][["Deaths","Cases","Medical doctors per 1000 people","Hospital beds per 1000 people"]].reset_index().drop(columns="index")#daily deaths
            dd2 = proc.loc[:len(proc.index)-1][["Deaths","Cases","Medical doctors per 1000 people","Hospital beds per 1000 people"]].reset_index().drop(columns="index")#daily deaths
            frames = [pd.DataFrame(data=[proc.loc[0]]), dd1-dd2]
            dailyDeaths = pd.concat(frames, join="inner").reset_index().drop(columns="index").loc[:len(proc.index)-1]["Deaths"]
            dailyDeaths.clip(lower=0,inplace=True)

            dailyCases = pd.concat(frames, join="inner").reset_index().drop(columns="index").loc[:len(proc.index)-1]["Cases"]
            dailyCases.clip(lower=1,inplace=True)#because some days cases were 0...

            dailyDoctors = pd.concat(frames, join="inner").reset_index().drop(columns="index").loc[:len(proc.index)-1]["Medical doctors per 1000 people"]
            dailyBeds = pd.concat(frames, join="inner").reset_index().drop(columns="index").loc[:len(proc.index)-1]["Hospital beds per 1000 people"]
            dailyBeds.loc[:].replace(to_replace=0.0,value=1,inplace=True)#because some days cases were 0...
            try :
                total.add(dailyCases)
            except :
                pass
            finally:
                total=dailyCases
        #plt.legend()
        plt.plot(range(len(proc.loc[:]["Date"])),total,label=str(proc.loc[0]["Entity"]))
        plt.title(i)
'''

if __name__ == "__main__":
    main()
