# Objective of this code : This code has summary of DataFrame
# Author: Raushan Kumar

# In Python, a DataFrame is a two-dimensional labeled data structure with columns of potentially different types. It is similar to a spreadsheet or SQL table, and is a fundamental data structure in data science.

# let's read data from pandas
import pandas as pd

# Reading CSV data
path = "C:\\Users\\Raushan\\Downloads\\projects\\public_projects\\Data-Science\\4. DataFrame\\sample-data\\"
data = pd.read_csv(path + "kc_housingdata.csv")

# fetching first five rows in this DataFrame
data.head(5)

# Reading Excel data
data = pd.read_excel(path + "kc_housingdata.xlsx")

# read dataframe in special encoding like - "utf-8"
data = pd.read_excel(path + "kc_housingdata.xlsx", encoding = 'utf-8')

# exporting dataframe in csv
data.to_csv(path + "kc_housin_exported.csv", index = False)

# exporting dataframe in Excel
data.to_excel(path + "kc_housin_exported.xlsx", index = False)

# exporting dataframe to clipboard
data.to_clipboard(index = False)

# pandas supports all major file type reading/writing
# Read more file type reading here

# https://pandas.pydata.org/docs/reference/io.html  
