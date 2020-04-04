"""
Data Normalization

An important technique to understand in data pre-processing.
When we take a look at the used car data set, we notice in the data that the feature “length” ranges from 150 to 250, while feature “width” and “height” ranges from 50 to 100.
We may want to normalize these variables so that the range of the values is consistent.
This normalization can make some statistical analyses easier down the road.
By making the ranges consistent between variables, normalization enables a fairer comparison between the different features.
Making sure they have the same impact, it is also important for computational reasons.

Another example:

--------------------        -------- ------------
|  Not-normalized  |        |  Normalized data  |
--------- ----------        --------- -----------
| age    | income  |        | age    | income   |
|--------|---------|        |--------|----------|
| 20     | 100000  |        | 0.2    | 0.2      |
| 30     | 20000   |        | 0.3    | 0.04     |
| 40     | 500000  |        | 0.4    | 1        |
--------- ----------        --------- -----------

Methods of normalizing data:
(*) Simple Feature scaling [0 ... 1] : x_new = x_old / x_max
(*) Min-Max [0 ... 1] : x_new = (x_old - x_min) / (x_old - x_min)
(*) Z-score [-3 ... 3] : x_new = (x_old - Mu) / sigma
    with:   Mu = average of the 
            sigma: standard deviation
"""

# importing the pandas module
import pandas as pd


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

# try to print the data before normalization
print(df["length"])

# Appliying Simple Feature Scaling to the length column
df["length"] = df["length"] / df["length"].max()

# Min-Max Scaling
df["length"] = (df["length"] - df["length"].min()) / (df["length"].max() - df["length"].min())

# Z-score Scaling
df["length"] = (df["length"] - df["length"].mean()) / df["length"].std()

# print the dataframe to see the new formatted data
print(df["length"])