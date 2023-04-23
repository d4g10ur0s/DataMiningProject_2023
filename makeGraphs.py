# Graphs
import seaborn as sn
import matplotlib.pyplot as plt
# Processing
import pandas as pd
import numpy as np

def graphByContinent(continents, data):
    for continent in continents :#Gia to idio continent 8a paei sto idio graph ?
        temp = data.loc[:][data["Continent"]==continent]
        heatmap = sn.heatmap(temp.iloc[:][["Cases",
                                           "Deaths",
                                           "Daily tests",
                                           "Hospital beds per 1000 people",
                                           "Medical doctors per 1000 people"]].cov(), annot=True, fmt='g')
        heatmap.set_title(continent)
        plt.savefig("Graphs\\"+continent+"_Heatmap"+".png")
        plt.clf()

        barplots = sn.catplot(data=pd.DataFrame(data=[temp.iloc[:][["Cases"]].mean(axis=0),
                                                     temp.iloc[:][["Deaths"]].mean(axis=0),
                                                     temp.iloc[:][["Daily tests"]].mean(axis=0),
                                                     temp.iloc[:][["Hospital beds per 1000 people"]].mean(axis=0)],), kind="bar",)
        barplots.set(title = continent)
        plt.savefig("Graphs\\"+continent+"_BarPlot"+".png")
        plt.clf()
