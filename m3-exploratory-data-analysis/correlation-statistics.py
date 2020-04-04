"""
Correlation Statistics

One way to measure the strength of the correlation between continuous numerical variable is by using a method called Pearson correlation.
Pearson correlation method will give you two values: the correlation coefficient and the P-value.
So how do we interpret these values?

"""

# importing the pandas module
import pandas as pd

# importing the stats module from scipy package
from scipy import stats


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

"""
we want to look at the correlation between the variable's horsepower and car price.
See how easy you can calculate the Pearson correlation using the SI/PI stats package?
We can see that the correlation coefficient is approximately.8, and this is close to 1.
So there is a strong positive correlation.
We can also see that the P-value is very small, much smaller than.001.
And so we can conclude that we are certain about the strong positive correlation.
"""
pearson_coeif, p_value = stats.pearsonr(df['horsepower'], df['price'])

print(pearson_coeif, p_value)