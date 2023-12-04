import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from pathlib import Path
%matplotlib inline
import os
# set to show decimal places up to 5
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# set the maximum number of rows and columns to display
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)


df = pd.read_csv(str(Path().resolve().parent) + "\\4. DataFrame\\sample-data\\weather.csv")

# Now, I will explore the data to gain insights about the data. 
# view dimensions of dataset
df.shape
df.head()
df.columns

# view summary of dataset
df.info()
# We can see that the data type of `Date` variable is object. I will parse the date currently coded as object into datetime format.

# parse the dates, currently coded as strings, into datetime format
df['Date'] = pd.to_datetime(df['Date'])
# extract year from date
df['Year'] = df['Date'].dt.year
# extract month from date
df['Month'] = df['Date'].dt.month
# extract day from date
df['Day'] = df['Date'].dt.day
# again view the summary of dataset
# We can see that there are three additional columns created from `Date` variable. Now, I will drop the original `Date` variable from the dataset.

# drop the original Date variable
df.drop('Date', axis=1, inplace = True)
# Now, I will explore the categorical variables one by one. 
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']
# check for missing values in categorical variables 
df[categorical].isnull().sum()
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method
pd.DataFrame(df.isna().sum() * 100 /df.shape[0])
# ### Explore Numerical Variables
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']
df[numerical].isnull().sum()

# We can see that all the 16 numerical variables contain missing values.
# Outliers in numerical variables
# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')


# The above boxplots confirm that there are lot of outliers in these variables.
# Check the distribution of variables
# Now, I will plot the histograms to check distributions to find out if they are normal or skewed. If the variable follows normal distribution, then I will do `Extreme Value Analysis` otherwise if they are skewed, I will find IQR (Interquantile range).

# plot histogram to check distribution

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')

# We can see that all the four variables are skewed. So, I will use interquantile range to find outliers.
# find outliers for Rainfall variable

for num_var in numerical:

    IQR = df[num_var].quantile(0.75) - df[num_var].quantile(0.25)
    Lower_fence = df[num_var].quantile(0.25) - (IQR * 3)
    Upper_fence = df[num_var].quantile(0.75) + (IQR * 3)
    print('{num_var} outliers are values < {lowerboundary} or > {upperboundary}'.format(num_var = num_var, lowerboundary=Lower_fence, upperboundary=Upper_fence))


X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']


# Split data into separate training and test set
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# check the shape of X_train and X_test
X_train.shape, X_test.shape
 
# Feature Engineering is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.

for col in numerical:
    if df[col].isnull().mean()>0:
        print(col, round(df[col].isnull().mean(),4))


# ### Assumption
# I assume that the data are missing completely at random (MCAR). There are two methods which can be used to impute missing values. One is mean or median imputation and other one is random sample imputation. When there are outliers in the dataset, we should use median imputation. So, I will use median imputation because median imputation is robust to outliers.

# I will impute missing values with the appropriate statistical measures of the data, in this case median. Imputation should be done over the training set, and then propagated to the test set. It means that the statistical measures to be used to fill missing values both in train and test set, should be extracted from the train set only. This is to avoid overfitting.

for col in numerical:
    col_median=df[col].median()
    df[col].fillna(col_median, inplace=True)           
      
# check again missing values in numerical variables in df
df[numerical].isnull().sum()
# Now, we can see that there are no missing values in the numerical columns.

# ### Engineering missing values in categorical variables
# print percentage of missing values in the categorical variables in training set
# print categorical variables with missing data
df[categorical].isnull().mean()

# impute missing categorical variables with most frequent value
for cat_col in categorical:
    df[cat_col].fillna(df[cat_col].mode()[0], inplace=True)

# check missing values in categorical variables in X_train
df[categorical].isnull().sum()

# ### Engineering outliers in numerical variables

# We have seen that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns contain outliers. I will use top-coding approach to cap maximum values and remove outliers from the above variables.

def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for num_var in numerical:
    df[num_var] = max_value(df, num_var, 3.2)

X_train.Rainfall.max(), X_test.Rainfall.max()


X_train.Evaporation.max(), X_test.Evaporation.max()


X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()


X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()


X_train[numerical].describe()


# We can now see that the outliers in `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns are capped.


# ### Encode categorical variables


categorical


X_train[categorical].head()


# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


X_train.head()


# We can see that two additional variables `RainToday_0` and `RainToday_1` are created from `RainToday` variable.
# 
# Now, I will create the `X_train` training set.


X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)


X_train.head()


# Similarly, I will create the `X_test` testing set.


X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)


X_test.head()


# We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called `feature scaling`. I will do it as follows.


# # **11. Feature Scaling** <a class="anchor" id="11"></a>
# 
# 
# [Table of Contents](#0.1)


X_train.describe()


cols = X_train.columns


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



X_train = pd.DataFrame(X_train, columns=[cols])


X_test = pd.DataFrame(X_test, columns=[cols])


X_train.describe()


# We now have `X_train` dataset ready to be fed into the Logistic Regression classifier. I will do it as follows.


# # **12. Model training** <a class="anchor" id="12"></a>
# 
# 
# [Table of Contents](#0.1)


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)



# # **13. Predict results** <a class="anchor" id="13"></a>
# 
# 
# [Table of Contents](#0.1)


y_pred_test = logreg.predict(X_test)

y_pred_test


# ### predict_proba method
# 
# 
# **predict_proba** method gives the probabilities for the target variable(0 and 1) in this case, in array form.
# 
# `0 is for probability of no rain` and `1 is for probability of rain.`


# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]


# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]


# # **14. Check accuracy score** <a class="anchor" id="14"></a>
# 
# 
# [Table of Contents](#0.1)


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# Here, **y_test** are the true class labels and **y_pred_test** are the predicted class labels in the test-set.


# ### Compare the train-set and test-set accuracy
# 
# 
# Now, I will compare the train-set and test-set accuracy to check for overfitting.


y_pred_train = logreg.predict(X_train)

y_pred_train


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# ### Check for overfitting and underfitting


# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))


# The training-set accuracy score is 0.8476 while the test-set accuracy to be 0.8501. These two values are quite comparable. So, there is no question of overfitting. 
# 


# In Logistic Regression, we use default value of C = 1. It provides good performance with approximately 85% accuracy on both the training and the test set. But the model performance on both the training and test set are very comparable. It is likely the case of underfitting. 
# 
# I will increase C and fit a more flexible model.


# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)


# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))


# We can see that, C=100 results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.


# Now, I will investigate, what happens if we use more regularized model than the default value of C=1, by setting C=0.01.


# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)


# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))


# So, if we use more regularized model by setting C=0.01, then both the training and test set accuracy decrease relatiev to the default parameters.


# ### Compare model accuracy with null accuracy
# 
# 
# So, the model accuracy is 0.8501. But, we cannot say that our model is very good based on the above accuracy. We must compare it with the **null accuracy**. Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.
# 
# So, we should first check the class distribution in the test set. 


# check class distribution in test set

y_test.value_counts()


# We can see that the occurences of most frequent class is 22067. So, we can calculate null accuracy by dividing 22067 by total number of occurences.


# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# We can see that our model accuracy score is 0.8501 but null accuracy score is 0.7759. So, we can conclude that our Logistic Regression model is doing a very good job in predicting the class labels.


# Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.
# 
# 
# But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making. 
# 
# 
# We have another tool called `Confusion matrix` that comes to our rescue.


# # **15. Confusion matrix** <a class="anchor" id="15"></a>
# 
# 
# [Table of Contents](#0.1)
# 
# 
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
# 
# 
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
# 
# 
# **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
# 
# 
# **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
# 
# 
# **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**
# 
# 
# 
# **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**
# 
# 
# 
# These four outcomes are summarized in a confusion matrix given below.
# 


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# The confusion matrix shows `20892 + 3285 = 24177 correct predictions` and `3087 + 1175 = 4262 incorrect predictions`.
# 
# 
# In this case, we have
# 
# 
# - `True Positives` (Actual Positive:1 and Predict Positive:1) - 20892
# 
# 
# - `True Negatives` (Actual Negative:0 and Predict Negative:0) - 3285
# 
# 
# - `False Positives` (Actual Negative:0 but Predict Positive:1) - 1175 `(Type I error)`
# 
# 
# - `False Negatives` (Actual Positive:1 but Predict Negative:0) - 3087 `(Type II error)`


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # **16. Classification metrices** <a class="anchor" id="16"></a>
# 
# 
# [Table of Contents](#0.1)


# ## Classification Report
# 
# 
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.
# 
# We can print a classification report as follows:-


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))


# ## Classification accuracy


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))



# ## Classification error


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))



# ## Precision
# 
# 
# **Precision** can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP). 
# 
# 
# So, **Precision** identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.
# 
# 
# 
# Mathematically, precision can be defined as the ratio of `TP to (TP + FP).`
# 
# 
# 


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))



# ## Recall
# 
# 
# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes.
# It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). **Recall** is also called **Sensitivity**.
# 
# 
# **Recall** identifies the proportion of correctly predicted actual positives.
# 
# 
# Mathematically, recall can be given as the ratio of `TP to (TP + FN).`
# 
# 
# 
# 


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# ## True Positive Rate
# 
# 
# **True Positive Rate** is synonymous with **Recall**.
# 


true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# ## False Positive Rate


false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# ## Specificity


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# ## f1-score
# 
# 
# **f1-score** is the weighted harmonic mean of precision and recall. The best possible **f1-score** would be 1.0 and the worst 
# would be 0.0.  **f1-score** is the harmonic mean of precision and recall. So, **f1-score** is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of `f1-score` should be used to 
# compare classifier models, not global accuracy.
# 
# 


# ## Support
# 
# 
# **Support** is the actual number of occurrences of the class in our dataset.


# # **17. Adjusting the threshold level** <a class="anchor" id="17"></a>
# 
# 
# [Table of Contents](#0.1)


# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob


# ### Observations
# 
# 
# - In each row, the numbers sum to 1.
# 
# 
# - There are 2 columns which correspond to 2 classes - 0 and 1.
# 
#     - Class 0 - predicted probability that there is no rain tomorrow.    
#     
#     - Class 1 - predicted probability that there is rain tomorrow.
#         
#     
# - Importance of predicted probabilities
# 
#     - We can rank the observations by probability of rain or no rain.
# 
# 
# - predict_proba process
# 
#     - Predicts the probabilities    
#     
#     - Choose the class with the highest probability    
#     
#     
# - Classification threshold level
# 
#     - There is a classification threshold level of 0.5.    
#     
#     - Class 1 - probability of rain is predicted if probability > 0.5.    
#     
#     - Class 0 - probability of no rain is predicted if probability < 0.5.    
#     
# 


# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df


# print the first 10 predicted probabilities for class 1 - Probability of rain

logreg.predict_proba(X_test)[0:10, 1]


# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg.predict_proba(X_test)[:, 1]


# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')


# ### Observations
# 
# 
# - We can see that the above histogram is highly positive skewed.
# 
# 
# - The first column tell us that there are approximately 15000 observations with probability between 0.0 and 0.1.
# 
# 
# - There are small number of observations with probability > 0.5.
# 
# 
# - So, these small number of observations predict that there will be rain tomorrow.
# 
# 
# - Majority of observations predict that there will be no rain tomorrow.


# ### Lower the threshold


from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')


# ### Comments
# 
# 
# - In binary problems, the threshold of 0.5 is used by default to convert predicted probabilities into class predictions.
# 
# 
# - Threshold can be adjusted to increase sensitivity or specificity. 
# 
# 
# - Sensitivity and specificity have an inverse relationship. Increasing one would always decrease the other and vice versa.
# 
# 
# - We can see that increasing the threshold level results in increased accuracy.
# 
# 
# - Adjusting the threshold level should be one of the last step you do in the model-building process.


# # **18. ROC - AUC** <a class="anchor" id="18"></a>
# 
# 
# [Table of Contents](#0.1)
# 
# 
# 
# ## ROC Curve
# 
# 
# Another tool to measure the classification model performance visually is **ROC Curve**. ROC Curve stands for **Receiver Operating Characteristic Curve**. An **ROC Curve** is a plot which shows the performance of a classification model at various 
# classification threshold levels. 
# 
# 
# 
# The **ROC Curve** plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold levels.
# 
# 
# 
# **True Positive Rate (TPR)** is also called **Recall**. It is defined as the ratio of `TP to (TP + FN).`
# 
# 
# 
# **False Positive Rate (FPR)** is defined as the ratio of `FP to (FP + TN).`
# 
# 
# 
# 
# In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various threshold levels. So, an ROC Curve plots TPR vs FPR at different classification threshold levels. If we lower the threshold levels, it may result in more items being classified as positve. It will increase both True Positives (TP) and False Positives (FP).
# 
# 


# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()



# ROC curve help us to choose a threshold level that balances sensitivity and specificity for a particular context.


# ## ROC-AUC
# 
# 
# **ROC AUC** stands for **Receiver Operating Characteristic - Area Under Curve**. It is a technique to compare classifier performance. In this technique, we measure the `area under the curve (AUC)`. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5. 
# 
# 
# So, **ROC AUC** is the percentage of the ROC plot that is underneath the curve.


# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# ### Comments
# 
# 
# - ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier.
# 
# - ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.


# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


# # **19. k-Fold Cross Validation** <a class="anchor" id="19"></a>
# 
# 
# [Table of Contents](#0.1)


# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))


# We can summarize the cross-validation accuracy by calculating its mean.


# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))


# Our, original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.


# # **20. Hyperparameter Optimization using GridSearch CV** <a class="anchor" id="20"></a>
# 
# 
# [Table of Contents](#0.1)


from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)



# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))


# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))


# ### Comments
# 
# 
# - Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507.
# 
# 
# - We can see that GridSearch CV improve the performance for this particular model.


# # **21. Results and conclusion** <a class="anchor" id="21"></a>
# 
# 

# 1.	The logistic regression model accuracy score is 0.8501. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.
# 
# 2.	Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.
# 
# 3.	The model shows no signs of overfitting.
# 
# 4.	Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.
# 
# 5.	Increasing the threshold level results in increased accuracy.
# 
# 6.	ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.
# 
# 7.	Our original model accuracy score is 0.8501 whereas accuracy score after RFECV is 0.8500. So, we can obtain approximately similar accuracy but with reduced set of features.
# 
# 8.	In the original model, we have FP = 1175 whereas FP1 = 1174. So, we get approximately same number of false positives. Also, FN = 3087 whereas FN1 = 3091. So, we get slighly higher false negatives.
# 
# 9.	Our, original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.
# 
# 10.	Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507. We can see that GridSearch CV improve the performance for this particular model.
# 

