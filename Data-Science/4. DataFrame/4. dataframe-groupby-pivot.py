# Let's learn how to query data from dataframe
# Importing Library
import pandas as pd
import numpy as np

df = pd.read_csv("sample-data\\kc_housingdata.csv")

# Get the counts of column
df.groupby('grade').size()

# Get aggrigation of a column at grouped level
df.groupby("grade", as_index=False)['price'].agg({np.max, np.min})

# Get aggrigation of multiple columns at grouped level
df.groupby('grade', as_index=False).agg({'price': np.max, 'sqft_living': np.mean})

# let's learn use of lambda in groupy
df_floor_share = df.groupby(["floors", "bedrooms"])["price"].mean().reset_index()
df_floor_share["floor_wise_avg_share"] = df_floor_share.groupby(["floors"])["price"].transform(lambda x : x/x.sum())
# filter df with mean price for each floor-bedroom greater than 1000
df.groupby(["floors", "bedrooms"]).filter(lambda x: x['price'].mean() > 1000)

# use of lambda in group by
df_smm = df.groupby("bedrooms").agg({"price": 'sum', "sqft_living": 'sum'}).reset_index()
df_smm["per_sqr_ft_cost"] = df.groupby('bedrooms').apply(lambda x: x['price'].sum() / x['sqft_living'].sum())

################### Pivot Table ###################
# Create pivot table and generate summary
df.pivot_table(values=["price"], index=["bedrooms"], columns = ["floors"],aggfunc = np.mean)
pd.pivot_table(df, values=["price"], index=["bedrooms"], columns = ["floors"],aggfunc = np.mean).reset_index(drop=True)
df.pivot_table(values=['price'], index=['bathrooms', 'bedrooms'], columns = ["floors"],aggfunc = [np.mean, np.sum])
