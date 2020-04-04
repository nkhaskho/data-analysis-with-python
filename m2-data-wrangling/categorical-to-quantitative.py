"""
Turning categorical into quantitative variables

In the car dataset, the "fuel-type" feature as a categorical variable has two values,
"gas" or "diesel”, which are in String format.
For further analysis, and in this example we will convert these variables into some form of numeric format.

"""

# importing the pandas module
import pandas as pd
import numpy as np


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

# pd.get_dummies() method gets the fuel-type column and creates the dataframe"dummy_variable_0".
print(pd.get_dummies(df['fuel-type']))