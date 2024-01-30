# AI_CourseWork
Repository to keep track of my AI coursework between home and campus. <br>
The data processing and implementation/evaluation metrics and desisions were laregly predefined by the assignment brief, including handling non-numerical and missing data, the training/test data split, the models we need to implement and more.

## Data Processing
- Non-Numerical Features: remove from dataset
- Missing data: Replace null values with the average for the column
- Splitting: 80/20 split for training/testing

### Removing Non-Numerical
The process of removing the non numical columns was simple, as we can just itterate through each column in the dataframe and add the names of non-nummerical columns to a list of strings.<br>
This list of strings is itterated through, and each one is dropped from the dataset.
```py
non_numeric_columns: list[str] = df.select_dtypes(
    exclude=np.number).columns.tolist()
```

### Missing Data
Replacing missing data with the mean for it's given column proved relatively easy, using the ```dataframe.fillna(value)``` method, as this automatically replaces all NaN values to the given value. <br>
Pandas also has a built in ```mean()``` method, that returns the mean of the column, so we can implement ```df = df.fillna(df.mean())```

### Splitting/Partitioning Data
There are multiple options for splitting the dataframe, inlcuding the ```train_test_split``` method. <br>
However, the easiest ways for this application is to use ```iloc[:]```, along with ```df.len()*0.8``` to determine how many indicies should be included/excluded in the new partition.


## Implementation and Evaluation
- Linear Regression:
-- Train using training set
-- Evaluate model using test set and compute: Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error
- Polynomial Regression:
-- Model degree of 2
-- Evaluate with same metrics
- Neural Network:
-- Define architecture, activation function, and optimised method
-- Evaluate as before

### Linear Regression
This is easily defined by ```LinearModel = LinearRegression()``` and adding the data by ```linearModel.fit(df_x_train,df_y_train)```. <br>
From here, the predictions are made ```x_poly = linearModel.predict(x_test)```. <br>
The MAE, MSE and RMSE can be calculated by SKlearn.metrics, ```print("MAE: ", metrics.mean_absolute_error(x_poly, df_y_test)```.<br>
<br>
