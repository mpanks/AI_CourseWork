import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data from file
df=pd.read_csv('Data.csv', skip_blank_lines=True)

# extract all data from Data.csv, except non-numerical features
#Bsmt SF 1&2 and unf SF removed - covered by Total Bsmt SF
#Bsmt Full&Half bath removed
#Garage cars removed - covered by garage area
#GarageYrBlt removed - unimportant
#Misc Val removed - ambiguity
#Bedroom AbvGr & Kitched AbvGr removed - covered by TotRms AbvGrd

FML: list[str] = [
    "Order",
    "Yr Sold",
    "Mo Sold",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
]
df.drop(FML, axis=1)
non_numeric_columns: list[str] = df.select_dtypes(
    exclude=np.number).columns.tolist()

df = df.drop(non_numeric_columns, axis = 1)
#Replace null values with average for the column

df = df.fillna(df.mean())

# Scale columns except last (sale price)
scaler=MinMaxScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

#Partition dataset 80/20 for training & testing
#Training data = 0-2344
#Test data = 2343-2929 
df_train: pd.DataFrame = df.iloc[: int(0.8*len(df))]
df_test: pd.DataFrame = df.iloc[int(0.8*len(df)):]

df_x_train: pd.DataFrame = df_train.drop(["SalePrice"],axis=1)
df_y_train: pd.DataFrame = df_train["SalePrice"]

df_x_test: pd.DataFrame = df_test.drop(["SalePrice"],axis=1)
df_y_test: pd.DataFrame = df_test["SalePrice"]


#Linear Regression
linearModel = LinearRegression()
linearModel.fit(df_x_train, df_y_train)

linPredict = linearModel.predict(df_x_test)
#fig=sns.regplot(x=df_y_test, y=linPredict)
plt.scatter(df_y_test,linPredict)
plt.plot(df_y_test, np.poly1d(np.polyfit(df_y_test,linPredict,1))(df_y_test),color="red")
#fig.set(xlabel='True Value',ylabel='Predicted Value')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("./plots/linearRegressionPredictions.png")
plt.clf()

##Compute Mean Absolute Error
print("Linear MAE: ", metrics.mean_absolute_error(df_y_test, linPredict))
##Compute Mean Squared Error
print("Linear MSE: ", metrics.mean_squared_error(df_y_test,linPredict))
##Compute Root Mean Squared Error
print("Linear RMSE: ", np.sqrt(metrics.mean_squared_error(df_y_test, linPredict)))

#Polynomial Regresssion

#Create and add data to model
polyRegModel = LinearRegression()
poly = PolynomialFeatures(degree=2,include_bias=False)
features = poly.fit_transform(df_x_train)

polyRegModel.fit(df_x_train,df_y_train)
polyPredict = polyRegModel.predict(df_x_test)

#Computations
print("Poly MAE: ",metrics.mean_absolute_error(df_y_test,polyPredict))
print("Poly MSE: ", metrics.mean_squared_error(df_y_test,polyPredict))
print("Poly RMSE: ",np.sqrt(metrics.mean_squared_error(df_y_test,polyPredict)))

#Create and save figure

plt.scatter(df_y_test, polyPredict)
plt.plot(df_y_test, np.poly1d(np.polyfit(df_y_test,polyPredict,1))(df_y_test),color="red")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("./plots/PolyRegression")
plt.clf()

#Neural Network
df = pd.concat([df_train,df_test])
df_train = df.iloc[:int(df.shape[0]*0.6)]
df_val = df.iloc[int(df.shape[0]*0.6):int(df.shape[0]*0.8)]
df_test = df.iloc[int(df.shape[0]*0.8) :]

df_x_train: pd.DataFrame = df_train.drop(["SalePrice"],axis=1)
df_y_train: pd.DataFrame = df_train["SalePrice"]

df_x_test: pd.DataFrame = df_test.drop(["SalePrice"],axis=1)
df_y_test: pd.DataFrame = df_test["SalePrice"]

df_x_val: pd.DataFrame = df_val.drop(["SalePrice"],axis=1)
df_y_val: pd.DataFrame = df_val["SalePrice"]

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

network = Sequential([
    Dense(128, activation='relu', input_shape=(38,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128,activation='relu'),
    Dense(1, activation='linear'),
])

optimizer = Adam(learning_rate=0.001)

network.compile(optimizer= optimizer,
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

hist = network.fit(df_x_train, df_y_train,
          batch_size=8, epochs=128,
          validation_data=(df_x_val, df_y_val),
          verbose = 0)

NNetPredictions: np.ndarray = network.predict(df_x_test)

#Loss of NNet
plt.clf()
plt.title("NNet Loss")
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
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig("./plots/NNetAccuracy.png")
plt.clf()

# Plot predictions against true values 
plt.scatter(df_y_test, NNetPredictions)
y_pred = NNetPredictions.flatten()
plt.plot(df_y_test,np.poly1d(np.polyfit(df_y_test,y_pred,1))(df_y_test), color="red")
plt.savefig("./plots/NNetPredictions.png")
plt.clf()

# Evaluate the model on the test set
test_loss, test_accuracy = network.evaluate(df_x_test, df_y_test, verbose=1)
# Print the accuracy
print(f"Test MSE: {test_accuracy}")
print(f"Val Loss: {test_loss}")
