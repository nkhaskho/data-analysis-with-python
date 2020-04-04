"""
Model evaluation using visualization.

Regression plots are a good estimate of:
* The relationship between two variables, 
* The strength of the correlation,
* And the direction of the relationship (positive or negative).

"""

# used modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

# using regplot function
sns.regplot(x="highway-mpg", y="price", data=df)

# using residplot function
sns.residplot(df["highway-mpg"], df["price"])

# add a delimieter on the y-axis
plt.ylim(0,)

# plot it
plt.show()
