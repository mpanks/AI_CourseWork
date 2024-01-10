import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Load data from file
allData=pd.read_csv('Data.csv', skip_blank_lines=True)

# extract all data from Data.csv, except non-numerical features
#Bsmt SF 1&2 and unf SF removed - covered by Total Bsmt SF
#Bsmt Full&Half bath removed
#Garage cars removed - covered by garage area
#GarageYrBlt removed - unimportant
#Misc Val removed - ambiguity
#Bedroom AbvGr & Kitched AbvGr removed - covered by TotRms AbvGrd
trimmedData = allData[["MS SubClass","Lot Frontage","Lot Area","Overall Qual","Overall Cond",
"Year Built","Year Remod/Add","Mas Vnr Area",
"Total Bsmt SF","1st Flr SF","2nd Flr SF",
"Low Qual Fin SF","Gr Liv Area",
"Full Bath","Half Bath",
"TotRms AbvGrd","Fireplaces","Garage Area",
"Wood Deck SF","Open Porch SF", "Enclosed Porch",
"3Ssn Porch","Screen Porch","Pool Area","Mo Sold","Yr Sold","SalePrice"]]
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

X, Y = trimmedData[["MS SubClass","Lot Frontage","Lot Area","Overall Qual","Overall Cond",
"Year Built","Year Remod/Add","Mas Vnr Area",
"Total Bsmt SF","1st Flr SF","2nd Flr SF",
"Low Qual Fin SF","Gr Liv Area",
"Full Bath","Half Bath",
"TotRms AbvGrd","Fireplaces","Garage Area",
"Wood Deck SF","Open Porch SF", "Enclosed Porch",
"3Ssn Porch","Screen Porch","Pool Area","Mo Sold","Yr Sold"]], trimmedData["SalePrice"]

#Linear Regression
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)
linearModel = LinearRegression()
linearModel.fit(xTrain, yTrain)

linPredict = linearModel.predict(xTest)
fig=sns.regplot(x=yTest, y=linPredict)
fig.figure.savefig("./plots/linearRegressionPredictions.png")
fig.clear()

##Compute Mean Absolute Error
print("Linear MAE: ", metrics.mean_absolute_error(yTest, linPredict))
##Compute Mean Squared Error
print("Linear MSE: ", metrics.mean_squared_error(yTest,linPredict))
##Compute Root Mean Squared Error
print("Linear RMSE: ", np.sqrt(metrics.mean_squared_error(yTest, linPredict)))

#Polynomial Regresssion

#Create and add data to model
poly = PolynomialFeatures(degree=2,include_bias=True)
features = poly.fit_transform(X)
xTrain, xTest, yTrain, yTest = train_test_split(features, Y, test_size=0.2, random_state=42)

polyRegModel = LinearRegression()
polyRegModel.fit(xTrain,yTrain)
polyPredict = polyRegModel.predict(xTest)

#Computations
print("Poly MAE: ",metrics.mean_absolute_error(yTest,polyPredict))
print("Poly MSE: ", metrics.mean_squared_error(yTest,polyPredict))
print("Poly RMSE: ",np.sqrt(metrics.mean_squared_error(yTest,polyPredict)))

#Create and save seaborn figure
fig = sns.regplot(x=yTest,y=polyPredict)
fig.figure.savefig("./plots/PolyRegression")
fig.clear()

#Neural Network
X_new, X_test, y_new, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
dev_per = X_test.shape[0]/X_new.shape[0]
X_train, X_dev, y_train, y_dev = train_test_split(X_new, y_new, test_size=dev_per, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(128, activation='relu', input_shape=(26,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128,activation='relu'),
    Dense(1, activation='linear'),
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer= optimizer,
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

hist = model.fit(X_train, y_train,
          batch_size=0, epochs=128,
          validation_data=(X_dev, y_dev),
          verbose = 0)


#Loss of NNet
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig("./plots/NNetLoss.png")
plt.clf()

#Accuracy of NNet
import matplotlib.pyplot as plt
plt.plot(hist.history['mean_squared_error'])
plt.plot(hist.history['val_mean_squared_error'])
plt.title('Model accuracy')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.savefig("./plots/NNetAccuracy.png")
plt.clf()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
# Print the accuracy
print(f"Test MSE: {test_accuracy}")
print(f"Val Loss: {test_loss}")

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
