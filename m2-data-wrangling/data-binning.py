"""
Data Binning

Binning as a method of data pre-processing.
Binning is when you group values together into bins. For example, you can bin “age” into [0 to 5], [6 to 10], [11 to 15] and so on.
Sometimes, binning can improve accuracy of the predictive models.
In addition, sometimes we use data binning to group a set of numerical values into a smaller number of bins to have a better understanding of the data distribution.
As example, “price” here is an attribute range from 5,000 to 45,500.
Using binning, we categorize the price into three bins: low price, medium price, and high prices.

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

# try to print the data before normalization
print(df["length"])

# creating the 3 bins, dividing price range to 3 interval
bins = np.linspace(min(df["price"]), max(df["price"]), 4)

# define group names
group_names = ['low', 'medium', 'hight']

df["price-binned"] = pd.cut(df["price"], bins, labels=group_names, include_lowest=True)

# print the dataframe to see the new formatted data
print(df["price-binned"])