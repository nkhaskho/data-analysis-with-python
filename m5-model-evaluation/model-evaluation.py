"""
Model Evaluation

Model Evaluation tells us how our model preforms in the real world.
In the previous module, we talked about the in-sample evaluation.
In-sample evaluation tells us how well our model fits the data already given to train it.
It does not give us an estimate of how well the trained model can predict new data.
The solution is to split our data up, use the In-sample data or training data to train the model.
The rest of the data called test data is used as out-of-sample data.
This data is then used to approximate how the model preforms in the real world.
Separating data into training and testing sets is an important part of model evaluation.
We use the test data to get an idea how our model will perform in the real world.
When we split a data set, usually the larger portion of data is used for training and a smaller part is used for testing.

"""

# used modules
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# get data file from url
# dataFile = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

# or just from a local repository
data_file = './data/imports-85.data'

# defining headers
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

# read file into pandas dataframe
df = pd.read_csv(data_file)

# set the dataframe header
df.columns = headers

# features or independent variables
x_data = df[['horsepower']]

# dataset target
y_data = df['price']

"""
split data into random train and test subsets
x_train, y_train: parts of available data as training set
x_test, y_test: parts of available data as testing set
test_size: percentage of the data for testing (eg 0.3 for 30%)
random_state: number generator used for random sampling

"""
x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.3,
    random_state=0
)

# print(x_train, x_test, y_train, y_test, end='\n\n')

# create a new LinearRegression instance
lr = LinearRegression()

# The cross_val_score() function returns a score value to tell us the cross-validation.
# Here, cv = 3, which means the data set is split into 3 equal partitions.
scores = cross_val_score(lr, x_data, y_data, cv=3)

print(scores)
"""
OUTPUT
[0.54831117 0.24849894 0.46083795]
"""



# FUNCTION CROSS_VAL_PREDICT()
"""
What if we want a little more information:
what if we want to know the actual predicted values supplied by our model before the R squared values are calculated?
To do this, we use the cross_val_predict() function.
The input parameters are exactly the same as the cross_val_score() function, but
the output is a prediction.
"""

yhat = cross_val_predict(lr, x_data, y_data)

# print(yhat)

"""
Let's illustrate the process.
First, we split the data into three folds; we use two folds for training, the remaining fold for testing.
The model will produce an output, and we will store it in and array.
We will repeat the process using two folds for training, one for testing.
The model produces an output again.
Finally, we use the last two folds for training, then we use the testing data.
This final testing fold produces an output.
These predictions are stored in an array.
"""




# Let's use R-squared to see if our assumption is correct.
"""
The closer the R-squared is to 1, the more accurate the model is.
Here we see the R-squared is optimal when the order of the polynomial is three.
The R-squared drastically decreases when the order is increased to 4, validating our initial
assumption.
We can calculate different R-squared values as follows:

"""

# First, we create an empty list to store the values.
rsqu_test = []

# We create a list containing different polynomial orders.
order = [1, 2, 3, 4]

# We then iterate through the list using a loop
for n in order:

    # We create a polynomial feature object with the order of the polynomial as a parameter
    pr = PolynomialFeatures(degree=n)

    # We transform the training and test data into a polynomial using the fit transform method
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])

    # We fit the regression model using the transformed data
    lr.fit(x_train, y_train)

    # then calculate the R-squared using the test data and store it in the array
    rsqu_test.append(lr.score(x_test, y_test))

print(rsqu_test)