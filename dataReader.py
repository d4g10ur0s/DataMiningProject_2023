# Graphs
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Processing
import pandas as pd
import numpy as np
# SVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

import math
import os

from makeGraphs import graphByContinent
from kmeans_cosinesimilarity import my_kmeans

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
    '''
    Part A.
    * plot countries per continent *
        1. plot moving averages of deaths divided by cases .
        2. plot covariance matrix of each country for columns , cases , deaths and daily tests .
    #3. create statistic graphs
    #graphs for continents and graphs for covariances
    graphByContinent(continents, countries ,data)
    '''

    '''
    Part B.
            * classify countries into categories *
            1. use mean values of deaths/cases and cases/daily_tests
            2. make a dataframe with each country as a column vector
            3. use kmeans with 2 centroids
    #4. make the new dataframe
    ks = None
    statistics = pd.DataFrame()
    mv = {}
    for i in countries :
        country = data.iloc[:][data["Entity"]==i]
        #a. normalize by max value of dataset and create statistics
        #a.1 normalize by max values of dataset

        country["Cases"] = country.iloc[:]["Cases"]/country.iloc[:]["Population"]
        country["Cases"] = country["Cases"]/country["Cases"].max()

        country["Deaths"] = country.iloc[:]["Deaths"]/country.iloc[:]["Population"]
        country["Deaths"] = country["Deaths"]/country["Deaths"].max()

        country["Daily tests"] = country.iloc[:]["Daily tests"]/data.iloc[:]["Population"]
        country["Daily tests"] = country.iloc[:]["Daily tests"]/data.iloc[:]["Daily tests"].max()

        country["Medical doctors per 1000 people"] = country.iloc[:]["Medical doctors per 1000 people"]/data.iloc[:]["Median age"]/(data.iloc[:]["Medical doctors per 1000 people"]/data.iloc[:]["Median age"]).max()
        country["Hospital beds per 1000 people"] = country.iloc[:]["Hospital beds per 1000 people"]/data.iloc[:]["Median age"]/(data.iloc[:]["Hospital beds per 1000 people"]/data.iloc[:]["Median age"]).max()

        #country["Median age"] = country.iloc[:]["Median age"]/data.iloc[:]["Population"]/(data.iloc[:]["Median age"]/data.iloc[:]["Population"]).max()

        #b. drop columns that dont matter
        toProcess=country.drop(columns=["Date",
                                        "Average temperature per year",
                                        "Longitude",
                                        "Latitude",
                                        "GDP/Capita",
                                        "Continent",
                                        "Population",
                                        "Median age",
                                        #"Medical doctors per 1000 people",
                                        #"Hospital beds per 1000 people",
                                        "Population aged 65 and over (%)",
                                        "Entity",
                                        #"Daily tests",
                                        ],inplace=False)
        #c. add to statistics
        mv[i] = toProcess.mean()
        statistics=pd.concat([statistics,toProcess])
        ks=toProcess.keys()
    #5. call kmeans
    print(ks)
    a = pd.DataFrame(data=mv).transpose().reset_index().drop(columns=["index"]).transpose().reset_index().drop(columns=["index"])
    for i in [50,100,150,200,250,500]:
        centroids ,classes = my_kmeans(a,maxIterations=i)
        b = a.transpose()
        #classes.append(1)
        indx = len(b.index)
        b[indx] = classes
        class_1 = b.loc[b[indx] > 0]
        class_2 = b.loc[b[indx] < 0.5]
        #Plotting the results
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(class_1[2] , class_1[3] , class_1[4] , color = 'red')
        ax.scatter(class_2[2] , class_2[3] , class_2[4] , color = 'black')
        #print("class 4")
        #print(class_1)
        #print("class 4")
        #print(class_2)
        ax.set_xlabel(ks[2])
        ax.set_ylabel(ks[3])
        ax.set_zlabel(ks[4])

        plt.savefig("Graphs\\kMeansGraphs\\"+str(i)+"-Means-"+ks[2]+" "+ks[3]+" "+ks[4]+".png")
        plt.clf()
    #print(a.iloc[5][:].max())
    '''

    '''
    Part C.
            * predict future percentage *
    '''
    #create X
    X = data.iloc[:][data["Entity"]=="Greece"].drop(columns=["Date",
                                                             "Average temperature per year",
                                                             "Longitude",
                                                             "Latitude",
                                                             "GDP/Capita",
                                                             "Continent",
                                                             "Medical doctors per 1000 people",
                                                             "Hospital beds per 1000 people",
                                                             "Population aged 65 and over (%)",
                                                             "Entity",
                                                             "Median age",
                                                             ],inplace=False)

    # normalize by max values of dataset
    X["Cases"] = X["Cases"]/X["Cases"].max()
    X["Deaths"] = X["Deaths"]/X["Deaths"].max()
    X["Daily tests"] = X.iloc[:]["Daily tests"]/data.iloc[:]["Daily tests"].max()
    # create train
    y = (X.iloc[:]["Cases"]/X.iloc[:]["Population"]).reset_index(inplace=False,drop=True)
    X.drop(columns=["Population"],inplace=True)
    y.pop(0)
    y_test = y.pop(len(y))
    y.fillna(y.mean())
    # SVM model with the RBF kernel
    model = SVR(kernel='rbf')
    model.fit(X.iloc[1:len(X)-1].values, y.values.tolist())
    # Make predictions on test data and calculate accuracy
    y_pred = model.predict(X.iloc[-1].values.reshape(1, -1))
    accuracy = math.sqrt((y_test - y_pred)**2)
    print('Accuracy:', accuracy)




if __name__ == "__main__":
    main()
