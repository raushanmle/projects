
# In this module we are going to learn most common plots from popular libraries
# Importing Library
import pandas as pd
from pathlib import Path

df = pd.read_csv(str(Path().resolve().parent) + "\\4. DataFrame\\sample-data\\kc_housingdata.csv")

################# Pandas Plots #########################
df.boxplot(column="price")
df[['bedrooms','bathrooms']].plot.line()
df.hist(column="bedrooms",by="price",bins=2)

import matplotlib.pyplot as plt
count, bins = np.histogram(df['price'],bins=5)
plt.hist(df['price'],bins=5,color='gray',edgecolor='white')

################# Matplot Library Plots #################


################# Seaborn Plots #########################

################# Plotly Plots ##########################