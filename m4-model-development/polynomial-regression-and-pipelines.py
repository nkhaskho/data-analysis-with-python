"""
Polynomial Regression and Pipelines.

What do we do when a linear model is not the best fit for our data?
Let’s look into another type of regression model: the polynomial regression.
We Transform our data into a polynomial, then use linear regression to fit the parameter.
Then we will discuss pipelines.
Pipelines are a way to simplify your code.
Polynomial regression is a special case of the general linear regression.
This method is beneficial for describing curvilinear relationships.
What is a curvilinear relationship?
It’s what you get by squaring or setting higher-order terms of the predictor variables in the model, transforming the data.
The model can be quadratic, which means that the predictor variable in the model is squared.

"""

# First we import all the modules we need.
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


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


# CALCULATE A POLYNOMIAL OF 3RD ORDER
print("\n\n\n# CALCULATE A POLYNOMIAL OF 3RD ORDER\n")

x = df["price"]
y = df["engine-size"]

# create a Polynomial of 3rd order
f = np.polyfit(x,y,3)

# 
p = np.poly1d(f)

# print the polynomial
print(p)
"""
OUTPUT
            3             2
-1.759e-12 x + 1.655e-07 x + 0.0002803 x + 94.56
"""


# POLYNOMIAL REGRESSION WITH THE ONE DIMENSION
print("\n\n\n# POLYNOMIAL REGRESSION WITH THE ONE DIMENSION\n")

# We use the "preprocessing" library in sci-kit-learn, to create a polynomial feature object.
pr = PolynomialFeatures(degree=2, include_bias=False)

# Then we transform the features into a polynomial feature with the “fit_transform” method.
x_polly = pr.fit_transform(df[['horsepower', 'curb-weight']])

print(x_polly)
"""
OUTPUT
[[1.1100000e+02 2.5480000e+03 1.2321000e+04 2.8282800e+05 6.4923040e+06]
 [1.5400000e+02 2.8230000e+03 2.3716000e+04 4.3474200e+05 7.9693290e+06]
 [1.0200000e+02 2.3370000e+03 1.0404000e+04 2.3837400e+05 5.4615690e+06]
 ...
 [1.3400000e+02 3.0120000e+03 1.7956000e+04 4.0360800e+05 9.0721440e+06]
 [1.0600000e+02 3.2170000e+03 1.1236000e+04 3.4100200e+05 1.0349089e+07]
 [1.1400000e+02 3.0620000e+03 1.2996000e+04 3.4906800e+05 9.3758440e+06]]
"""



# PREPROCESSING
print("\n\n\n# PREPROCESSING\n")

# We can use the preprocessing module to simplify many tasks.

# We import “StandardScaler”
from sklearn.preprocessing import StandardScaler

# We train the object
SCALE = StandardScaler()

# fit the scale object;
SCALE.fit(df[['horsepower', 'highway-mpg']])

# then transform the data into a new dataframe on array “x_scale”.
x_scale = SCALE.transform(df[['horsepower', 'highway-mpg']])

print(x_scale)
"""
OUTPUT
[[ 0.19101743 -0.54779597]
 [ 1.24419463 -0.69311506]
 [-0.029415   -0.11183871]
 [ 0.2889874  -1.27439141]
 [ 0.16652494 -0.83843415]
            ...
 [ 0.75434477 -1.12907233]
 [ 0.06855497 -0.54779597]
 [ 0.26449491 -0.83843415]]
"""


# PIPELINES
print("\n\n\n# PIPELINES\n")

"""
There are more normalization methods available in the preprocessing library, as well as other transformations.
We can simplify our code by using a pipeline library.
There are many steps to getting a prediction, for example, Normalization, Polynomial transform, and Linear regression.
We simplify the process using a pipeline.
Pipelines sequentially perform a series of transformation.
The last step carries out a prediction.
"""


# We create a list of tuples, the first element in the tuple contains the name of the estimator: model. The second element contain model constructor.
Input = [
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2)),
    ('model', LinearRegression)
]

# We input the list in the pipeline constructor.
Pipe = Pipeline(Input)

print(Pipe)

# We now have a pipeline object, We can train the pipeline by applying the train method to the Pipeline object.
# Pipe.train(df['horsepower', 'curb-weight', 'engine-size', 'highway-mpg'], y)

# We can also produce a prediction as well.
# yhat = Pipe.predict(df['horsepower', 'curb-weight', 'engine-size', 'highway-mpg'])
