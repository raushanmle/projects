# Creating dummy variables is a common step in statistical modeling and machine learning. It's used when you have categorical variables (variables that represent different categories or groups) and you want to include these in your model.

# Most algorithms work with numerical data and don't know how to handle categorical data. By creating dummy variables, you convert each category into a new column, and assign a 1 or 0 (True/False) value to the column. This process is also known as one-hot encoding.

# For example, if you have a categorical variable "Color" with values "Red", "Blue", and "Green", you would create three new variables (dummy variables): "Color_Red", "Color_Blue", and "Color_Green". For a row in your dataset that has "Color" = "Red", the "Color_Red" variable would be 1 and the other two variables would be 0.

# Importing Library
import pandas as pd
from pathlib import Path

df = pd.read_csv(str(Path().resolve().parent) + "\\4. DataFrame\\sample-data\\titanic-dataset.csv")

# Create dummy variables
df_dummies = pd.get_dummies(df, columns=['Embarked'], dtype= int)
