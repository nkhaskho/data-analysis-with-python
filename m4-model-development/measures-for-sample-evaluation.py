"""
Sample Evaluation

Now that we’ve seen how we can evaluate a model by using visualization, we want to numerically evaluate our models.
Let’s look at some of the measures that we use for in-sample evaluation.
These measures are a way to numerically determine how good the model fits on our data.
Two important measures that we often use to determine the fit of a model are: Mean Square Error (MSE), and R-squared.
To measure the MSE, we find the difference between the actual value y and the predicted value yhat then square it.

"""

# First we import all the modules we need.
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


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


# MEAN SQUARED ERROR (MSE)
print("\n\n\n# MEAN SQUARED ERROR (MSE)\n")

# Select predict
Y_predict_simple_fit = df['horsepower']

print(mean_squared_error(df['price'], Y_predict_simple_fit))
"""
OUTPUT
229771966.04411766
"""


# R-SQUARED (R^2)
"""
Coefficient of Determination or R^2 is 1 minus the ratio of the MSE of the regression line divided by the MSE of the average of the data points.
For the most part, it takes values between 0 and 1.
"""
print("\n\n\n# R-SQUARED (R^2)\n")

# create a Linear Regression object
lm = LinearRegression()

# Select X and Y
X = df[['highway-mpg']]
Y = df['price']

# fir the model
lm.fit(X, Y)

# get score
print(lm.score(X, Y))

"""
OUTPUT
0.4733503228708666
From the value that we get from this example, we can say that approximately 47.335% of the variation of price is explained by this simple linear model.
Your R^2 value is usually between 0 and 1, if your R^2 is negative it can be due to overfitting
"""