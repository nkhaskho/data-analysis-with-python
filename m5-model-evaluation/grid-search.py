"""
Grid Search

Grid search allows us to scan through multiple free parameters with few lines of code.
Parameters like the alpha term discussed in the previous video are not part of the fitting or training process.
These values are called hyperparameters.
Scikit-learn has a means of automatically iterating over these hyperparameters using cross-validation.
This method is called Grid search.
Grid search takes the model or objects you would like to train and different values of the hyperparameters.
It then calculates the mean square error or R squared for various hyperparameter values, allowing you to choose the best values.

"""

# used modules
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


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

# We create a ridge regression object or model
ridge = Ridge()

# dictionary of parameter values.
parameters = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]

# then create a Grid Search CV object
# the inputs are the ridge regression object, the parameter values and the number of folds.
# We will use R squared; this is the default scoring method.
grid = GridSearchCV(ridge, parameters, cv=4)

# We fit the object
grid.fit(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], df['price'])

# We can find the best values for the free parameters using the attribute best estimator
print(grid.best_estimator_)
"""
OUTPUT
Ridge(alpha=10000, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
"""

# We can also get information like the mean score on the validation data using the attribute cv result.
scores = grid.cv_results_
print(scores['mean_test_score'])
"""
OUTPUT
[0.62523347 0.62523497 0.62524866 0.62538441 0.62664023 0.63331572
 0.63675224 0.60678915]
"""