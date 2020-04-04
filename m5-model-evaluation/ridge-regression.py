"""
Ridge Regression.

Ridge regression prevents over-fitting.
In this section we will focus on polynomial regression for visualization,
 but over-fitting is also a big problem when you have multiple independent variables or features.
 
"""

# used modules
import pandas as pd
from sklearn.linear_model import Ridge


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

# Create a Ridge object using the constructor
"""
For alpha equal to 0.01, the estimated function tracks the actual function.
When alpha equals 1, we see the first signs of under-fitting. The estimated function does not have enough flexibility.
At alpha equals to 10, we see extreme under-fitting; it does not even track the two points.
"""
RidgeModel = Ridge(alpha=0.1)

# features or independent variables
X = df[['horsepower', 'highway-mpg', 'engine-size']]

# Defining data target
Y = df['price']

# We train the model using the fit method
RidgeModel.fit(X, Y)

# To make a prediction, we use the predict method
Yhat = RidgeModel.predict(X)

# print the prediction result
print(Yhat)

