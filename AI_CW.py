import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data from file
f=pd.read_csv('C:\\Users\\matth\\Downloads\\Data.csv')
exampleData = f[["Lot Area","SalePrice"]]

# exampleGraph = sns.load_dataset(exampleData)


#calculatorStuff = exampleData["Lot Area"].to_numpy()

# Splits 'exampledata' dict into two lists
tempLotArea = exampleData["Lot Area"]
tempSalePrice = exampleData["SalePrice"]
landVal = []

# calculates landvalue, area/price, adds to list
for index in range(len(tempLotArea)):
    lotArea = tempLotArea[index]
    salePrice = tempSalePrice[index]
    tempVal = lotArea/salePrice
    landVal.append(tempVal)

# Creates numpy list, calculates mean and std
numpyLandVal = np.array(landVal)
mean = numpyLandVal.mean()
std = numpyLandVal.std()
removed = []

# itterates numpy list, removes outliers in dataset
for index in range(len(numpyLandVal)):
    if numpyLandVal[index]>=(mean+(3*std)):
        removed.append(index)
        exampleData = exampleData.drop(index)
    elif numpyLandVal[index]<=(mean-(3*std)):
       removed.append(index)
       exampleData = exampleData.drop(index)

# Outputs data as scatter graph
graph = sns.FacetGrid(exampleData)
graph.map(plt.scatter,"SalePrice","Lot Area",edgecolor = "Blue" ).add_legend()
plt.show()
