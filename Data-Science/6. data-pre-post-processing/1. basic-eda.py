# In this module, we are going to learn most common steps involved in EDA.

# Missing value treatment is an important step in data science, as missing values can affect the accuracy of statistical analyses and machine learning models. There are several methods for treating missing values, including:

# 1. **Deletion:** This involves removing rows or columns with missing values from the dataset. This method is simple but can result in loss of information and bias in the remaining data.

# 2. **Imputation:** This involves replacing missing values with estimated values based on the available data. There are several methods for imputing missing values, including mean imputation, median imputation, mode imputation, and regression imputation.

# 3. **Prediction:** This involves using machine learning models to predict missing values based on the available data. This method can be more accurate than imputation but requires more computational resources and may be more complex to implement.


# Importing Library
import pandas as pd
from pathlib import Path

df = pd.read_csv(str(Path().resolve().parent) + "\\4. DataFrame\\sample-data\\kc_housingdata.csv")