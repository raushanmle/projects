# Let's learn how to query data from dataframe
# Importing Library
import pandas as pd
import numpy as np

df = pd.read_csv("sample-data\\kc_housingdata.csv")

# Get the counts of column
df.groupby('grade').size()

# Get aggrigation of a column at grouped level
df.groupby("grade", as_index=False)['price'].agg({np.max, np.min})

# Get aggrigation of columns at grouped level
df.groupby('grade', as_index=False).agg({'price': np.max, 'sqft_living': np.mean})

# 
df.pivot_table(values=["price"], index=["bedrooms", "floors"], aggfunc=np.mean)
df.pivot_table(values=['price'], index=['bathrooms', 'bedrooms'], aggfunc=np.mean)

(df.groupby(['floors', 'bedrooms']).sum())
(df.groupby(['floors', 'bedrooms']).sum().loc[lambda df: df['waterfront'] == 15])
