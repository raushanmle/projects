# Data correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together. In other words, it's a way to understand the relationship between multiple variables and attributes in your dataset.

# There are several types of correlation coefficients, but the most commonly used are:
# 1. Pearson correlation coefficient (r): This measures the linear relationship between two variables. The coefficient ranges from -1 to 1. A value of 1 means a perfect positive correlation (as one variable increases, the other does too), -1 means a perfect negative correlation (as one variable increases, the other decreases), and 0 means no linear correlation.
# 2. Spearman rank correlation coefficient (ρ or rs): This measures the monotonic relationship between two variables, and is more robust to outliers. It's based on the ranked values for each variable rather than the raw data.
# 3. Kendall rank correlation coefficient (τ): This measures the ordinal association between two variables. It's used when you have ordinal data.

# Correlation can be a powerful tool for feature selection and understanding the relationships in your data, but it's important to remember that correlation does not imply causation. Just because two variables are correlated, it doesn't mean that changes in one variable are causing changes in another.



import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(str(Path().resolve().parent) + "\\4. DataFrame\\sample-data\\kc_housingdata.csv")

# Compute Pearson correlation

df.drop(['id', 'date'], axis=1, inplace= True)

pearson_corr = df.corr(method='pearson')

# Compute Spearman correlation
spearman_corr = df.corr(method='spearman')

# Compute Kendall correlation
kendall_corr = df.corr(method='kendall')

plt.figure(figsize=(10,8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm')
plt.title("Pearson Correlation Heatmap")
plt.show()