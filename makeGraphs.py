# Graphs
import seaborn as sn
import matplotlib.pyplot as plt
# Processing
import pandas as pd
import numpy as np
# Logging
import logging

def dataNormalization(columns , data):
    div = []
    for i in columns :
        data[i] = data.iloc[:][i]/data.iloc[:][i].max()

def getWeeklyMovingAverages(country):
    ma = 30
    movingAverages = []
    toProcess = ((country.iloc[:]["Deaths"])/(country.iloc[:]["Cases"]).replace(0,1))
    i = 0
    while i < (len(toProcess)):
        s=0
        for j in range(ma):
            if j+i >= len(toProcess):
                pass
            else:
                s+=toProcess.iloc[j+i]#/country.iloc[0]["Population"]
        ## end for
        movingAverages.append(s/ma)
        i+=1
    # end while
    return movingAverages

def createMovingAveragesGraphs(continents, data):
    for continent in continents :
        temp = data.iloc[:][data["Continent"]==continent]
        tempCountries = temp["Entity"].unique().tolist()
        movingAverages = []
        print("Weekly Moving Averages")
        for c in tempCountries:
            #print("Processing : " + c)
            movingAverages.append(getWeeklyMovingAverages(temp.iloc[:][temp["Entity"]==c]))
            #plt.plot(getWeeklyMovingAverages(temp.iloc[:][temp["Entity"]==c]), label = c)
        ## end for
        continentCountries = pd.DataFrame(data=movingAverages)#every line has a country
        for i in range(len(continentCountries)):
            plt.plot(continentCountries.iloc[i][:], label = tempCountries[i])
        plt.legend(loc='upper right')
        plt.title(continent)
        print("To save : " + continent)
        plt.savefig("Graphs\\"+continent+"_MovingAverages"+".png")
        #plt.show()
        plt.clf()
        ## end for
    # end for

def countryCovariance(country):
    toProcess=country.drop(columns=["Date",
                                    "Average temperature per year",
                                    "Longitude",
                                    "Latitude",
                                    "GDP/Capita",
                                    "Continent",
                                    "Entity",
                                    ],inplace=False)
                                    #"Population"
    dataNormalization(toProcess.keys(), toProcess)
    #toProcess['Cases'] = toProcess.loc[:]['Cases']/country.loc[:]['Population']
    #toProcess['Deaths'] = toProcess.loc[:]['Deaths']/country.loc[:]['Population']
    #toProcess['Daily tests'] = toProcess.loc[:]['Daily tests']/country.loc[:]['Population']
    #print(toProcess.keys())
    #input(toProcess.cov())
    #The only columns that matter are Daily tests , Cases and Deaths ,
    #that is because covariance with other columns is 0 .
    heatmap = sn.heatmap(toProcess.iloc[:][:].corr(method='spearman').fillna(0,inplace=False), annot=True, fmt='g')
    #plt.show()
    #heatmap.set_title(country.iloc[0]["Entity"])
    #print("To save : " + country.iloc[0]["Entity"])
    #plt.savefig("Graphs\\"+country.iloc[0]["Entity"]+"_CovarianceMatrix"+".png")
    heatmap.set_title(country.iloc[0]["Continent"])
    print("To save : " + country.iloc[0]["Continent"])
    plt.savefig("Graphs\\"+country.iloc[0]["Continent"]+"_CovarianceMatrixnoPopulation"+".png")
    plt.clf()

def graphByContinent(continents, countries, data):
    #create moving averages for each continent
    createMovingAveragesGraphs(continents , data)
    # create covariance matrix for each country
    #for country in continents :
    for country in continents :
        countryCovariance(data.iloc[:][data["Continent"] == country])
    return None
