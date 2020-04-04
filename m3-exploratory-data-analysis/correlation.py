"""
Correlation

Correlation is a statistical metric for measuring to what extent different variables are interdependent.
In other words, when we look at two variables over time, if one variable changes how does this affect change in the other variable?

"""

# importing the pandas module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

print(df[['horsepower','price']].corr())

"""
sns.regplot(x="horsepower", y="price", data=df)
plt.ylim(0,)
"""