# Graphs
import seaborn as sn
import matplotlib.pyplot as plt
# Processing
import pandas as pd
import numpy as np

def dataNormalization(columns , data):
    div = []
    for i in columns :
        data[i] = data.iloc[:][i]/data.iloc[:][i].max()


def getWeeklyMovingAverages(country):
    movingAverages = []
    toProcess = (country.iloc[:]["Deaths"]/country.iloc[:]["Cases"]).fillna(0)
    for i in range(len(toProcess)):
        s=0
        for j in range(7):
            if j+i >= len(toProcess):
                pass
            else:
                s+=toProcess.iloc[j+i]
        ## end for
        movingAverages.append(s/7)
    # end for
    return movingAverages

def createMovingAveragesGraphs(continents, data):
    for continent in continents :
        temp = data.iloc[:][data["Continent"]==continent]
        tempCountries = temp["Entity"].unique().tolist()
        movingAverages = []
        for c in tempCountries:
            movingAverages.append(getWeeklyMovingAverages(temp.iloc[:][temp["Entity"]==c]))
        ## end for
        continentCountries = pd.DataFrame(data=movingAverages).fillna(0) #every line has a country
        for i in range(len(continentCountries)):
            plt.plot(continentCountries.iloc[i][:], label = tempCountries[i])
        ## end for
        plt.legend()
        plt.title(continent)
        plt.savefig("Graphs\\"+continent+"_MovingAverages"+".png")
        plt.clf()
    # end for

def countryCovariance(country):
    toProcess=country.drop(columns=["Date",
                                    "Average temperature per year",
                                    "Longitude",
                                    "Latitude",
                                    "GDP/Capita",
                                    "Continent",
                                    "Entity"],inplace=False)
    dataNormalization(toProcess.keys(), toProcess)
    #print(toProcess)
    #input(toProcess.cov())
    #The only columns that matter are Daily tests , Cases and Deaths ,
    #that is because covariance with other columns is 0 .
    heatmap = sn.heatmap(toProcess.iloc[:][["Daily tests","Cases","Deaths"]].cov(), annot=True, fmt='g')
    heatmap.set_title(country.iloc[0]["Entity"])
    #plt.show()
    plt.savefig("Graphs\\"+country.iloc[0]["Entity"]+"_CovarianceMatrix"+".png")
    plt.clf()

def graphByContinent(continents, countries, data):
    #create moving averages for each continent
    #createMovingAveragesGraphs(continents , data)
    # create covariance matrix for each country
    for country in countries :
        countryCovariance(data.iloc[:][data["Entity"] == country])
