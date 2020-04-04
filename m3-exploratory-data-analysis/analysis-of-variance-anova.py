"""
Analysis of Variance (ANOVA)

ANOVA is a statistical test that stands for "Analysis of Variance".
ANOVA can be used to find the correlation between different groups of a categorical
variable.

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

# select the wanted columns
df_anova = df[["make", "price"]]

df_anova_grp = df_anova.groupby(["make"])

for key, item in df_anova_grp:
    print(df_anova_grp.get_group(key), "\n\n")
