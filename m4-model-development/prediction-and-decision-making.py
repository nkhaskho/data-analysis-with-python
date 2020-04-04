"""
Prediction and Decision Making

How can we determine if our model is correct?
The first thing you should do is make sure your model results make sense.
You should always use Visualization, Numerical measures for evaluation, and Comparing between different models.
Let’s look at an example of prediction; if you recall we train the model using the fit method.

"""

# First we import all the modules we need.
import pandas as pd
import numpy as np
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


# create a Linear Regression object
lm = LinearRegression()

# Select X and Y
X = df[['highway-mpg']]
Y = df['price']

# fit the model
lm.fit(X, Y)

# get score
print(lm.predict(30))
print(lm.coef_)

# generate a sequence from 1 to 100
new_input = np.arange(1, 101, 1).reshape(-1, 1)

# print(new_input)
print(lm.predict(new_input))