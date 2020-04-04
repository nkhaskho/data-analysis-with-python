"""
Getting Started Analysing Data

Introduce some simple pandas methods that all data scientists and analysts should know when using Python pandas and data.
At this point, we assume that the data has been loaded.
It's time for us to explore the dataset.
Panda's has several built in methods that could be used to understand the datatype of features or to look at the distribution of data within the dataset.
Using these methods gives an overview of the dataset.
And also point out potential issues, such as the wrong datatype of features, which may need to be resolved later on.
Data has a variety of types.
The main types stored in Pandas objects are object, float, int, and datetime.
The datatype names are somewhat different from those in native Python.
This table shows the differences and similarities between them.
Some are very similar, such as the numeric datatypes "int" and "float".
The "object" pandas type functions similar to "string" in Python, save for the change in name, while the "datetime" pandas type, is a very useful type for handling time series
data.
There are two reasons to check data types in a dataset.
Pandas automatically assigns types based on the encoding it detects from the original data table.
For a number of reasons, this assignment may be incorrect.

"""

# importing the pandas module
import pandas as pd


# get data file from url
# dataFile = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

# or just from a local repository
data_file = './data/imports-85.data'

# if we want to save dataframe into an output file
output_file = './data/cars.csv'

# defining headers
headers = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

# read file into pandas dataframe
df = pd.read_csv(data_file)

# set the dataframe header
df.columns = headers

# statistical summary of the data
print(df.describe(include="all"))

# check data types
print(df.dtypes)

# concice summary of the dataframe
print(df.info)