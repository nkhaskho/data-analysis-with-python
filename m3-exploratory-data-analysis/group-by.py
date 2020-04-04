"""
GroupBy method

The group by method is used on categorical variables, groups the data into subsets according
to the different categories of that variable.
You can group by a single variable or you can group by multiple variables by passing
in multiple variable names.

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

# select the wanted columns
df_test = df[["drive-wheels", "body-style", "price"]]

# group items
df_grp = df_test.groupby(["drive-wheels", "body-style"], as_index=False)

# print the df_grp dataframe
print(df_grp)