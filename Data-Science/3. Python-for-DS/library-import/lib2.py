# importing standard library
import pandas

# importing and assigning a small variable name
import pandas as pd
import numpy as np

# importing only required function
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import boxplot

# importing function from saved file
from lib1 import fn_division
fn_division(10, 4)


########## If there is any error follow these steps ##########

# check if path is already appended
import sys
sys.path
# check if your path if your working dir is available path

# append path if you don't see
sys.path.append('c:\\Users\\Raushan\\Downloads\\projects\\public_projects\\Data-Science\\3. Python-for-DS\\library-import')
# let's run same function
fn_division(10, 4)
