"""
Data Formatting

Problem of data with different formats, units, and conventions, and the pandas methods that help us deal with these issues.
Data is usually collected from different places, by different people, which may be stored in different formats.
Data formatting means bringing data into a common standard of expression that allows users to make meaningful comparisons.
As a part of dataset cleaning, data formatting ensures that data is consistent and easily understandable.

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

# convert "mpg" to "L/100km"
df["city-mpg"] = 235/df["city-mpg"]

# rename the dataframe column "city-mpg"
df.rename(columns={"city-mpg": 'city-L/100km'}, inplace=True)

# or just print the column to see it's details (Name, Length and dtype)
# print(df["city-L/100km"].tail(5))

# convert datatype to int in price column
df["price"] = df["price"].astype("int")

# print the dataframe to see the new formatted data
print(df)


