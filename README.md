# AI_CourseWork
Repository to keep track of my AI coursework between home and campus. <br>
The data processing and implementation/evaluation metrics and desisions were laregly predefined by the assignment brief, including handling non-numerical and missing data, the training/test data split, the models we need to implement and more.

## Data Processing
- Non-Numerical Features: remove from dataset
- Missing data: Replace null values with the average for the column
- Splitting: 80/20 split for training/testing

### Removing Non-Numerical
The process of removing the non numical columns was simple, as these just weren't taken from the "alldata" dataframe created when we read the csv file.

### Missing Data
Replacing missing data with the mean for it's given column proved relatively easy, using the dataframe.fillna(value) method, as this automatically replaces all NaN values to the given value. Pandas also has a built in mean() method, that returns the mean of the column.

### Splitting/Partitioning Data
There are multiple options for splitting the dataframe. However, the easiest ways for this application were to use iloc[:_numberOfIndexes] and tail(_numberOfIndexes) where _numberOfIndexes is the number of indexes to be included for the partition.
As there are 2930 total entries in the csv file, this means the test data receives the first 2344 entries, and the test data receives the last 586 entries.

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