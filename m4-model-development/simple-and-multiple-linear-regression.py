"""
Simple Linear regression and Multiple Linear Regression.

Linear Regression will refer to one independent variable to make a prediction.
Multiple Linear Regression will refer to multiple independent variables to make a prediction.

"""

# importing the pandas module
import pandas as pd

# from scikit-learn package importing LinearRegression class from linear_model module
from sklearn.linear_model import LinearRegression


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



# (1) SIMPLE LINEAR REGRESSION
# ________________________________

"""
Simple Linear Regression (or SLR) is: A method to help us understand the relationship between two variables: 
The predictor (independent) variable X,
And the target (dependent) variable Y.
We would like to come up with a linear relationship
"""

# create a Linear Regression object
lm = LinearRegression()

# define predictor and target variable
X = df[['highway-mpg']]
Y = df['price']


# fit the Model (used to find a and b parameters of linear relation Y = a.X + b)
# a = lm.coef_
# b = lm.intercept_
lm.fit(X, Y)

# obtain a prediction
Yhat = lm.predict(X)

# we can see the prediction array
print(Yhat[0])


# (2) MULTIPLE LINEAR REGRESSION
# ________________________________

"""
Multiple Linear Regression is used to 
explain the relationship between one continuous target (Y) variable, and two or more predictor (X) variables.
"""

# extract 4 predictor variables and store them in the variable Z
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# fit the model
lm.fit(Z, df['price'])

# obtain a prediction
mpredict_price = lm.predict(Z)