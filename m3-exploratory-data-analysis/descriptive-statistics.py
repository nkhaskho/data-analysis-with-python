"""
Descriptive Statistics

When you begin to analyze data, it’s important to first explore your data before you spend
time building complicated models. One easy way to do so is to calculate some
descriptive statistics for your data. Descriptive statistical analysis helps to
describe basic features of a dataset and obtains a short summary about the sample and measures
of the data.

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

"""
the describe() function automatically computes basic statistics for all numerical variables. 
It shows the mean, the total number of data points, the standard deviation, the quartiles and the extreme values.
Any NaN values are automatically skipped in these statistics. This function will give you a clearer idea.
"""
print(df.describe())


# summarize the categorical data (fwd, rwd, 4wd) by using the function value_counts()
drive_wheels_count = df['drive-wheels'].value_counts()
print(drive_wheels_count)

# show a boxplot using seaborn
sns.boxplot(x="drive-wheels", y="price", data=df)

# setting up x-axis and y-axis
y = df[["engine-size"]]
x = df["price"]


"""
We will thus plot the engine size on the x-axis and the price on the y-axis.
We are using the Matplotlib function “scatter” here, taking in x and a y variable.
"""
plt.scatter(x, y)

# setting up plot legends
plt.title("price = f(engine-size)")
plt.xlabel("engine-size")
plt.ylabel("price")

# show the plot
plt.show()
