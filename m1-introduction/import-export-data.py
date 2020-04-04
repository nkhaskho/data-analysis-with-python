"""
Importing and Exporting Data

Read in data using Python’s pandas package.
Once we have our data in Python, then we can perform all the subsequent data analysis procedures we need.
Data acquisition is a process of loading and reading data into notebook from various sources.

"""

# importing the pandas module
import pandas as pd


# get data file from url
# dataFile = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

# or just from your local computer
data_file = './data/imports-85.data'

# if you want to save your data into an output file
output_file = './data/exports-data.csv'

# defining the dataframe headers
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

# read file into pandas dataframe
# you can set header argument to None 
df = pd.read_csv(data_file)

# set the dataframe header
df.columns = headers

# try thoes methods
# head(n), tail(n), describe(), dtypes()
# see https://pandas.pydata.org/docs/ for more pandas documentations
print(df.head(5))

# save dataframe 
# We can export the dataframe to_csv(), to_excel(), to_json() or to_sql()
df.to_csv(output_file)
