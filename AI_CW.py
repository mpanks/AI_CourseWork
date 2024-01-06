import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Load data from file
allData=pd.read_csv('Data.csv', skip_blank_lines=True)

# extract all data from Data.csv, except non-numerical features
trimmedData = allData[["MS SubClass","Lot Frontage","Lot Area","Overall Qual","Overall Cond",
"Year Built","Year Remod/Add","Mas Vnr Area",
"BsmtFin SF 1","BsmtFin SF 2","Bsmt Unf SF","Total Bsmt SF","1st Flr SF","2nd Flr SF",
"Low Qual Fin SF","Gr Liv Area","Bsmt Full Bath","Bsmt Half Bath",
"Full Bath","Half Bath","Bedroom AbvGr","Kitchen AbvGr",
"TotRms AbvGrd","Fireplaces","Garage Yr Blt","Garage Cars","Garage Area",
"Wood Deck SF","Open Porch SF", "Enclosed Porch",
"3Ssn Porch","Screen Porch","Pool Area","Misc Val","Mo Sold","Yr Sold","SalePrice"]]
#print("Trimmed data:",trimmedData)

#Replace null values with average for the column

for col in trimmedData:

    currentCol = trimmedData[col]
    mean = float
    mean = currentCol.mean()
    trimmedData[col] = trimmedData[col].fillna(mean)

#Partition dataset 80/20 for training & testingpytho
#2930 indexes, highest index = 2929
#Training data = 0-2344
#Test data = 2343-2929 
trainingData = trimmedData.iloc[:2344]
testData = trimmedData.tail(586)

x_test= testData[["MS SubClass","Lot Frontage","Lot Area","Overall Qual","Overall Cond",
"Year Built","Year Remod/Add","Mas Vnr Area",
"BsmtFin SF 1","BsmtFin SF 2","Bsmt Unf SF","Total Bsmt SF","1st Flr SF","2nd Flr SF",
"Low Qual Fin SF","Gr Liv Area","Bsmt Full Bath","Bsmt Half Bath",
"Full Bath","Half Bath","Bedroom AbvGr","Kitchen AbvGr",
"TotRms AbvGrd","Fireplaces","Garage Yr Blt","Garage Cars","Garage Area",
"Wood Deck SF","Open Porch SF", "Enclosed Porch",
"3Ssn Porch","Screen Porch","Pool Area","Misc Val","Mo Sold","Yr Sold"]]

x_train=trainingData[["MS SubClass","Lot Frontage","Lot Area","Overall Qual","Overall Cond",
"Year Built","Year Remod/Add","Mas Vnr Area",
"BsmtFin SF 1","BsmtFin SF 2","Bsmt Unf SF","Total Bsmt SF","1st Flr SF","2nd Flr SF",
"Low Qual Fin SF","Gr Liv Area","Bsmt Full Bath","Bsmt Half Bath",
"Full Bath","Half Bath","Bedroom AbvGr","Kitchen AbvGr",
"TotRms AbvGrd","Fireplaces","Garage Yr Blt","Garage Cars","Garage Area",
"Wood Deck SF","Open Porch SF", "Enclosed Porch",
"3Ssn Porch","Screen Porch","Pool Area","Misc Val","Mo Sold","Yr Sold"]]

y_test= testData["SalePrice"]
y_train=trainingData["SalePrice"]

#Linear Regression

linearModel = LinearRegression()
linearModel.fit(x_train, y_train)

predictions = linearModel.predict(x_test)
plt.scatter(y_test, predictions)
plt.savefig("./plots/linearRegressionPredictions.png")
plt.clf()

##Compute Mean Absolute Error
print("Linear MAE: ", metrics.mean_absolute_error(y_test, predictions))

##Compute Mean Squared Error
print("Linear MSE: ", metrics.mean_squared_error(y_test,predictions))

##Compute Root Mean Squared Error
print("Linear RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#Polynomial Regresssion

#Neural Network


# exampleGraph = sns.load_dataset(exampleData)
#calculatorStuff = exampleData["Lot Area"].to_numpy()
# Splits 'exampledata' dict into two lists

#tempLotArea = exampleData["Lot Area"]
#tempSalePrice = exampleData["SalePrice"]
#landVal = []

# calculates landvalue, area/price, adds to list
#for index in range(len(tempLotArea)):
 #   lotArea = tempLotArea[index]
  #  salePrice = tempSalePrice[index]
   # tempVal = lotArea/salePrice
    #landVal.append(tempVal)

# Creates numpy list, calculates mean and std
#numpyLandVal = np.array(landVal)
#mean = numpyLandVal.mean()
#std = numpyLandVal.std()
#removed = []

# itterates numpy list, removes outliers in dataset
#for index in range(len(numpyLandVal)):
#    if numpyLandVal[index]>=(mean+(3*std)):
#        removed.append(index)
#        exampleData = exampleData.drop(index)
#    elif numpyLandVal[index]<=(mean-(3*std)):
#       removed.append(index)
#       exampleData = exampleData.drop(index)

# Outputs data as scatter graph
#graph = sns.FacetGrid(exampleData)
#graph.map(plt.scatter,"SalePrice","Lot Area",edgecolor = "Blue" ).add_legend()

# Generates line of best fit, adds to graph before display
#model = np.poly1d(np.polyfit(exampleData["SalePrice"],exampleData["Lot Area"],1))
#polyline = np.linspace(exampleData.min(), exampleData.max(), 50)
#plt.plot(polyline, model(polyline))
#plt.savefig("./plots/test.png")
#plt.clf()
