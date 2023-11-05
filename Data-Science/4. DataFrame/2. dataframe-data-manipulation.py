# Importing Library
import pandas as pd
import numpy as np

df = pd.read_excel("sample-df\\kc_housingdata.xlsx")

# We are going to learn basic operations on dataframe.
# Get the first 5 rows of the dataframe
df.head()
# Get the last 5 rows of the dataframe
df.tail()
# Get brief information about each columns
df.describe()
# Get the information about the dataframe
df.info()
# Get column names
df.columns
# Get data types of each column
df.dtypes
# Get the shape of the dataframe
df.shape
# Get NA values count in each column
df.isna().sum()
# Get null values count in each column
df.isnull().sum()

# Now Let's learn how to drop NA values
# Drops all rows that have any NaN values
df.dropna()
# Drop only if ALL columns are NaN   
df.dropna(how='all')
# Drops row if it does not have at least two 
df.dropna(thresh=2)
# Drops only if NaN in specific column (as asked in the question)
df.dropna(subset=[1])

# Let's learn how to fill NA values
# Fill NA values with 0
df.fillna(0)
# fill value just above from cell
df.fillna(method='ffill') 

# fill value with column df given in dictionary
values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df.fillna(value=values)
df.fillna(value=values, limit=1)
# fill value with mean of column
df.fillna(df.mean())
# fill value with median of column
df.fillna(df.median())
# fill value with mode of column
df.fillna(df.mode())

# KNNImputer is a class in the scikit-learn library that provides imputation for missing values using k-Nearest Neighbors. It replaces missing values with the mean value of the k-nearest neighbors in the feature space.
import numpy as np
from sklearn.impute import KNNImputer
X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform(X)

# set to show decimal places up to 5
pd.set_option('display.float_format', lambda x: '%.5f' % x)
# set the maximum number of rows and columns to display
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

# Get unique values of a columns
df['bedrooms'].unique()
# Get number of unique values of a columns
df['bedrooms'].nunique()
# Get count of each unique value of a columns
df['bedrooms'].value_counts()
# Get count of each unique value of a columns in percentage
df['bedrooms'].value_counts(normalize=True)
# Get percentile of column
np.percentile(df['price'], range(0,100,10))
# Binning a column
bined = [0,5,10,15,20,25,30,35]
grp_level = ['a','b','c','d','e','f','g']
pd.cut(df['bedrooms'], bined, labels = grp_level)
# Get quantile of column
quants = [0.05, 0.25, 0.5, 0.75, 0.95]
df.quantile(quants)
# sort values by column name
df.sort_values('floors', ascending=True)
# Get the index of min value of a column
df['price'].idxmin()
# Get the index of max value of a column
df['price'].idxmax()
