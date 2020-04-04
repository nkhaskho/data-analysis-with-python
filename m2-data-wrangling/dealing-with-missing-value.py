"""
Dealing with Missed Values

The pervasive problem of missing values, as well as strategies on what to do when you encounter missing values in your data.
When nodata value is stored for a feature for a particular observation, we say this feature has a “missing value”.
Usually “missing value: in dataset” appears as “?”, “N/A”, 0 or just a blank cell.

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


# let's modify a price of the row 0, which is already 16500
df['price'][0] = None

# the dropna method will delete all rows with NaN value
# here when you print the df, you will see that the first row of the dataframe was dropped
"""
subset: list of columns
axis: 0 to drop the entire row, and 1 to drop the entire column
inplace: to save result in the same dataframe
df.dropna(subset=['price'], axis=0) this line of code doesn't change the df 
"""
result_df = df.dropna(subset=['price'], axis=0, inplace=True)

# or just you can do as default
result_df = df.dropna()

print(result_df)

# we can also replace that value with the replace() method
# see the official documentation
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
print(df['price'].replace({
    None: '5',
    16500: 15501
}))

"""
OUTPUT
[203 rows x 26 columns]
1      15501.0
2      13950.0
3      17450.0
4      15250.0
5      17710.0
        ...   
199    16845.0
200    19045.0
201    21485.0
202    22470.0
203    22625.0
Name: price, Length: 203, dtype: float64
"""
