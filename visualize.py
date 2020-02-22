# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import style
style.use('ggplot')


def getValues(feature, df):
    if(feature == "Petal Length"):
        return df.iloc[0:150, 2].values
    elif(feature == "Petal Width"):
        return df.iloc[0:150, 3].values
    elif(feature == "Sepal Length"):
        return df.iloc[0:150, 0].values
    elif(feature == "Sepal Width"):
        return df.iloc[0:150, 1].values


def plot(xFeature, yFeature, df):
    x = getValues(xFeature, df)
    y = getValues(yFeature, df)
    plt.figure().canvas.set_window_title(yFeature + ' Vs ' + xFeature)
    plt.scatter(x[:50], y[:50], color='mediumseagreen',
                marker='x', label='setosa')
    plt.scatter(x[50:100], y[50:100], color='crimson',
                marker='d', label='versicolor')
    plt.scatter(x[100:150], y[100:150], color='dodgerblue',
                marker='+', label='virginica')
    plt.title(yFeature + ' Vs ' + xFeature)
    plt.xlabel(xFeature, color='black')
    plt.ylabel(yFeature, color='black')
    plt.legend(loc='lower right')
    plt.show()


# Read the input file
df = pd.read_csv("./data/iris.data", header=None)

# Plot the data
plot('Petal Width', 'Petal Length', df)
plot('Sepal Length', 'Petal Length', df)
plot('Sepal Width', 'Petal Length', df)
plot('Sepal Length', 'Petal Width', df)
plot('Sepal Width', 'Petal Width', df)
plot('Sepal Width', 'Sepal Length', df)
