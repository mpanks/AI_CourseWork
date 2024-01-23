import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# Load data from file
df: pd.DataFrame = pd.read_csv('Data.csv', skip_blank_lines=False)

#Remove unnecessary columns from DataFrame
useless_items: list[str] = [
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

df = df.drop(useless_items, axis=1)

#Create list of non-numeric columns
non_numeric_columns: list[str] = df.select_dtypes(
    exclude=np.number).columns.tolist()

#Remove non-numeric data
df = df.drop(non_numeric_columns, axis = 1)

#Replace null values with average for the column
df = df.fillna(df.mean())

#Scale columns except last (sale price)
scaler=MinMaxScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

#Partition dataset 80/20 for training & testing
df_train: pd.DataFrame = df.iloc[: int(0.8*len(df))]
df_test: pd.DataFrame = df.iloc[int(0.8*len(df)):]

df_x_train: pd.DataFrame = df_train.drop(["SalePrice"],axis=1)
df_y_train: pd.DataFrame = df_train["SalePrice"]

df_x_test: pd.DataFrame = df_test.drop(["SalePrice"],axis=1)
df_y_test: pd.DataFrame = df_test["SalePrice"]

def linearReg(df_x_train, df_x_test, df_y_train, df_y_test):
    from sklearn.linear_model import LinearRegression
    #Linear Regression
    linearModel = LinearRegression()
    linearModel.fit(df_x_train, df_y_train)
    linPredict = linearModel.predict(df_x_test)

    #Save graph to file
    plt.scatter(df_y_test,linPredict)
    plt.plot(df_y_test, np.poly1d(np.polyfit(df_y_test,linPredict,1))(df_y_test),color="red")
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

def polynomialReg(df_x_train, df_x_test, df_y_train, df_y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    #Polynomial Regresssion

    #Add data to model
    polyRegModel = LinearRegression()
    poly = PolynomialFeatures(degree=2,include_bias=False)
    features = poly.fit_transform(df_x_train)

    polyRegModel.fit(X=features,y=df_y_train)
    x_poly = poly.fit_transform(df_x_test)
    polyPredict = polyRegModel.predict(x_poly)

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

def NNet(df,df_x_train, df_x_test, df_y_train, df_y_test):
    from sklearn.model_selection import train_test_split
    
    #Neural Network
    #Partition dataset
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
    #Define network, add layers&neurons
    network = Sequential([
        Dense(110, activation='relu', input_shape=(24,)),
        Dense(110, activation='relu'),
        Dense(110, activation='relu'),
        Dense(110, activation='relu'),
        Dense(110,activation='relu'),
        Dense(64,activation='relu'),
        Dense(1, activation='linear'),
    ]   )

    optimizer = Adam(learning_rate=0.001)
    
    #Compile network
    network.compile(optimizer= optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    #Train modeel
    hist = network.fit(df_x_train, df_y_train,
              batch_size=18, epochs=250,
              validation_data=(df_x_val, df_y_val),
            verbose = 0)

    NNetPredictions: np.ndarray = network.predict(df_x_test)

    #Loss of NNet
    plt.title("NNet Loss")
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig("./plots/NNetLoss.png")
    plt.clf()

    #Plot predictions against true values 
    plt.scatter(df_y_test, NNetPredictions)
    y_pred = NNetPredictions.flatten()
    plt.plot(df_y_test,np.poly1d(np.polyfit(df_y_test,y_pred,1))(df_y_test), color="red")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Neural Network Predictions")
    plt.savefig("./plots/NNetPredictions.png")
    plt.clf()

    #Evaluate the model on the test set
    test_loss, test_accuracy = network.evaluate(df_x_test, df_y_test, verbose=1)
    #Print computations
    print("NNet MAE: ", metrics.mean_absolute_error(df_y_test,y_pred))
    print(f"NNet MSE: {test_accuracy}")
    print(f"NNet RMSE: {np.sqrt(test_loss)}")

def Further(df_x_train, df_x_test, df_y_train, df_y_test):
    #Source:
    #https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/
    from catboost import CatBoostRegressor
    
    #Define CatBoost regressor
    cb_model = CatBoostRegressor()
    
    #Fit data to model
    cb_model.fit(df_x_train, df_y_train, verbose=0)
    preds = cb_model.predict(df_x_test)
    
    #Print computed metrics
    print("CatBoost MAE: ", metrics.mean_absolute_error(df_y_test, preds))
    print("CatBoost MSE: ", metrics.mean_squared_error(df_y_test, preds))
    print("CatBoost RMSE: ", np.sqrt(metrics.mean_squared_error(df_y_test, preds)))
    
    #Create and save figure
    plt.scatter (df_y_test,preds)
    y_pred = preds.flatten()
    plt.plot(df_y_test,np.poly1d(np.polyfit(df_y_test,y_pred,1))(df_y_test), color="red")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("CatBoost Predictions")
    plt.savefig("./plots/CatBoostPredictions.png")
    plt.clf()
    
#Calls above functions, only added to make debugging easier/quicker
linearReg(df_x_train, df_x_test, df_y_train, df_y_test)
polynomialReg(df_x_train, df_x_test, df_y_train, df_y_test)
NNet(df,df_x_train, df_x_test, df_y_train, df_y_test)
Further (df_x_train, df_x_test, df_y_train,df_y_test)