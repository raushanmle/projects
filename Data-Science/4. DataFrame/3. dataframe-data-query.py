
# Let's learn how to query data from dataframe
# Importing Library
import pandas as pd
import numpy as np

df = pd.read_csv("sample-data\\kc_housingdata.csv")
# fetch columns from DF
df[['price', 'bedrooms']]
# Fetch rows with bedroom is equal to 3 
df[df["bedrooms"] == 3]
# Fetch rows with bedroom is equal to 3 and price is greater than 500000
df[(df["bedrooms"] == 3) & (df["price"] > 500000)]
# Fetch rows with bedroom is equal to 1 and 3
df[df["bedrooms"].isin([1, 3])]
df[(df["bedrooms"] == 3) | (df["bedrooms"] == 1)]
# Fetch rows with bedroom is not equal to 3
df[~(df["bedrooms"] == 3)]
df[df["bedrooms"] != 3]

# Select specific columns based on condition
df.loc[df["bedrooms"] != 3, ["price", "sqft_living"]]

# Show specific row
df.iloc[1]
# SHow specific range rows
df.iloc[3:5]
# SHow specific range rows & columns
df.iloc[3:5, 0:2]
# By lists of integer position locations, similar to the numpy/python style
df.iloc[[1, 2, 4], [0, 2]]
# Selecting specific columns and all rows
df.iloc[:, 1:3]
# Select based on multiple condition
conditions = [
    (df['bedrooms'] == 3), (df['bathrooms'] == 1),
    (df['floors'] == 1), (df['grade'] == 7)]
choices = ['3-bed', 'single', 'first-floor', "high-grade"]
df['type-of-house'] = np.select(conditions, choices, default='no-class')

# DataFrame merge
df_zip_code = df.drop_duplicates("zipcode")[["zipcode", "lat", "long"]].rename(columns={"lat": "lat_updated", "long": "long_updated"})
df_add_updated = pd.merge(df, df_zip_code, on = ["zipcode"], how = "left")
pd.merge(df, df_zip_code, on = ["zipcode"], how = "right")

pd.merge(df, df_zip_code, on = ["zipcode"], how = "inner")






import matplotlib.pyplot as plt
%matplotlib inline
df.boxplot(column="price")
df[['bedrooms','bathrooms']].plot.line()
df.hist(column="bedrooms",by="price",bins=2)
np.unique(df['bedrooms'])


import matplotlib.pyplot as plt
count, bins = np.histogram(df['price'],bins=5)
plt.hist(df['price'],bins=5,color='gray',edgecolor='white')


df['price'].quantile(q=[.8,.9])
y = list(x)[0]


df[df['price'] > y]

for i in df.columns:
    x = df[i].unique()

dict={}
for i in df.columns:
    dict[i] = len(df[i])
    dict[i] = df[i].unique()
    

((df['bathrooms'].append(df['bedrooms'])).unique())

df.loc[lambda df: df['bedrooms'] > 3, :]


df.head()





df.where(df.floors>0)


df.where(lambda df:df.floors == 1,lambda df: df['price'] + 1,axis = 'columns')


x=[1,2,3,4,5]


lambda y:y*y


z(x[3])


df['color'] = df.Set.map( lambda x: 'red' if x == 'Z' else 'green')


df.head()


np.unique(df['type_of_flat'])


df['type_of_flat']= ""


(df['bedrooms'] <= 3)


condition =[(df['bedrooms'] <= 3),((df['bedrooms']<=6) & (df['bedrooms']>3)),(df['bedrooms']>=7)]
type_of_room = ['small','medium','large']
df['type_of_flat']=np.select(condition,type_of_room,default='black')


df.groupby('type_of_flat').count()


#df = pd.DataFrame({'Type':list('ABBC'), 'Set':list('ZZXY')})
conditions = [
    (df['Set'] == 'Z') & (df['Type'] == 'A'),
    (df['Set'] == 'Z') & (df['Type'] == 'B'),
    (df['Type'] == 'B')]
choices = ['yellow', 'blue', 'purple']
df['color'] = np.select(conditions, choices, default='black')
print(df)


df['type_of_flat'].value_counts()




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df.head()
df1=df.drop('date',axis=1)


df1=pd.get_dummies(df1)


x=df1
y=df1['price']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=400)


x_train.shape,y_train.shape


y_train.head()


x_train.head()


reg = LinearRegression()


reg.fit(x_train.drop(['id','zipcode','price'],axis = 1),y_train)


reg.score(x_test.drop(['id','zipcode','price'],axis = 1), y_test)


reg.coef_


reg.intercept_


reg.singular_


x_data = x_test.drop(['id','zipcode'],axis =1)


x_predicted_data = pd.DataFrame(reg.predict(x_test.drop(['id','zipcode','price'],axis = 1)),index=None)


x_data = x_data.reset_index()


x_test['price'].reset_index()


pd.concat([x_data,x_predicted_data,x_test['price'].reset_index()],axis=1)


a1=x_test.drop(['id','zipcode','price'],axis =1).iloc[[0,1],:]


a1


reg.predict(a1)


df.info()


df3 = pd.read_csv("C:\\Users\\Raushan Kumar\\Downloads\IRIS.csv",header = None)


df3['result']=np.where(df3[8]=='setosa',1,0)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df3.head()


x=df3.drop(8,axis=1)
y=df3.result


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.20,random_state=50)


clf = LogisticRegression(penalty='l2',random_state=10).fit(x_train,y_train)


clf.predict(x_test)


y_test


clf.score(x_test,y_test)


df3


from statsmodels.api import OLS


from sklearn.metrics import classification_report


x_train


clf.predict(x_test)


print(classification_report(clf.predict(x_test), y_test))


from sklearn.metrics import confusion_matrix


confusion_matrix(y_test,clf.predict(x_test))


import statsmodels.api as sm


x=x_train.drop(['result'],axis=1)


x


logit_model=sm.Logit(y_train,x)


#result=logit_model.fit()


import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats


diabetes = datasets.load_diabetes()
X = diabetes.df
y = diabetes.target

lm = LinearRegression()
lm.fit(X,y)
params = np.append(lm.intercept_,lm.coef_)
predictions = lm.predict(X)


newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))


MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))


var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())


sd_b = np.sqrt(var_b)


ts_b = params/ sd_b


p_values =[2*(1-stats.t.cdf(np.abs(i),((len(newX)*newX.shape[1])-len(newX[0])))) for i in ts_b]


p_values


2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))


stats.t.cdf()


[(np.abs(i),i) for i in ts_b]


len(df3),len(df3[0])


len(newX),len(newX[0])


[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]






# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))





p_values =[2*(1-stats.t.cdf(np.abs(i),((len(newX)*newX.shape[1])-len(newX[0])))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
print(myDF3)


x= "abcs dfer"
y={'a','c'}
z=[]
[z.append(x.replace(i,"")) for i in y]


import re


t = 'as888-888-5587'


string1 = "Hello, world. hey word \n"


re.search(r".........", string1)


re.search(r"l+o", string1)


re.search(r"H.?e", string1) # 0 or 1 char in between H and e


Matches the preceding element zero or more times. For example, ab*c matches "ac", "abc", "abbbc", etc. [xyz]* matches "", "x", "y", "z", "zx", "zyx", "xyzzy", and so on. (ab)* matches "", "ab", "abab", "ababab", and so on.


re.search(r"e(ll)*o", string1) #(ab) matches ababab only in this pattern any times


re.search(r"^He", string1) #begins with
re.search(r"rld$", string1) #ends with


[abcx-z] matches "a", "b", "c", "x", "y", or "z", as does [a-cx-z].


re.search(r"[aeioul]+", string1)


re.search(r"[^abc]", string1)


re.search(r"....[d]", string1) #match 4 chars before "."


re.findall("@...........", s)


s = 'aaa@xxx.com bbb@yyy.com ccc@zzz.com ww.f333kart.com@'
print(re.sub('[a-z]*@', 'ABC@', s,3)) #substitute with value # last number how many time replacement shd be performed


print(re.sub('[xyz23]', '1', s)) # any item matched will be replaced


print(re.sub('aaa|bbb|ccc', 'ABC', s))


print(re.sub('([a-z]*)@', '\\1-123@', s))


t = re.subn('[a-z]*@', 'ABC@', s)
t[0]


re.sub('\d{3}', 'ABC', s) # replace that many times repeated digits to char
#re.sub('\d', 'ABC', s)


x=input()


var = x.split(" ")


j=0
for i in range(2):
    if int(var[i+1])-int(var[i])<0:
        j=j+1


x,y,z=(input()).split(" ")


x


name = input()

name2=input()



var = name2.split(" ")
j=0
for i in range(0,(int(name)-1)):

        if (i==0 and int(var[i+1])-int(var[i])<0):
                j=j+2

        elif int(var[i+1])-int(var[i])<0:
                j=j+1
print(j+1)


y=6


for i in range(1,int(y)):
    if i>1:
        for j in range(2,i):
            if (j%i)==0:
                break
        else:
            print(i)
 


lower = int(input("Enter lower range: "))  
upper = int(input("Enter upper range: "))  
k=[]
for j in range(1,upper + 1):

    if j > 1:  
        for i in range(2,j):  
            if (j % i) == 0:  
                break  
            else:  
                k.append(j)  
        


k=[2,3,4]


mul=1
for l in k:
        mul=mul*l


def mult(x):
    total =1
    for i in x:
        total=total*i
    return total


from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.df[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, test_idx=None,resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=colors[idx],marker=markers[idx], label=cl,edgecolor='black')
    # highlight test samples
    if test_idx:
    # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],c='', edgecolor='black', alpha=1.0,linewidth=1, marker='o',s=100, label='test set')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


print('Labels counts in y:', np.bincount(y))


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)


y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))


X_test_std.shape


X_combined_std.shape


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0],X_xor[y_xor == 1, 1],c='b', marker='x',label='1')
plt.scatter(X_xor[y_xor == -1, 0],X_xor[y_xor == -1, 1],c='r',marker='s',label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.show()


from sklearn.svm import SVC


svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()


svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,y_combined, classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


np.random.uniform(1.5,1.6,size =10)


>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.decomposition import PCA
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.pipeline import make_pipeline
>>> pipe_lr = make_pipeline(StandardScaler(),
... PCA(n_components=2),
... LogisticRegression(random_state=1))
>>> pipe_lr.fit(X_train, y_train)
>>> y_pred = pipe_lr.predict(X_test)
>>> print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


iris_data = pd.DataFrame(iris['df'],columns= ['sepal length (cm)',
  'sepal width (cm)',
  'petal length (cm)',
  'petal width (cm)'])


import seaborn as sn
import matplotlib.pyplot as plt


corr_mat = iris_data.corr()
sn.heatmap(corr_mat, annot=True)
plt.show()


Dict = {'A': [45,37,42,35,39],
        'B': [38,31,26,28,33],
        'C': [10,15,17,21,12]
        }


Dict['A']


corr_matrix = iris_data.corr()
corr_matrix['sepal length (cm)'].sort_values(ascending=False)


import os
import tarfile
from six.moves import urllib


housing = pd.read_csv("C:\\Users\\Raushan Kumar\\Downloads\housing.csv")


df.info()


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


corr_mat = housing.corr()
sn.heatmap(corr_mat, annot=True)
plt.show()


import numpy as np
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x=imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]) #we can column name in fit to replace with mean


x.transform([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])


np.mean([[7, 2, 3], [4, 0, 6], [10, 5, 9]])


housing['ocean_proximity'].isnull().sum()


Y=housing['median_house_value']
X=housing.drop('median_house_value',axis=1)


>>> from sklearn.preprocessing import LabelEncoder
>>> encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing['ocean_proximity'] = encoder.fit_transform(housing_cat)


>>> from sklearn.preprocessing import OneHotEncoder
>>> encoder = OneHotEncoder()
>>> housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
>>> housing_cat_1hot.toarray()


x=np.array([[0., 0., 0., 1., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 1., 0.]])


x.reshape(-1,1) #- means not sure of rows or columns


import collections
collections.Counter(housing_cat_encoded)


(collections.Counter(housing["ocean_proximity"]))


x=[1,2,3,4,4,4,6,7,5,6,6,6,9]
collections.Counter(x)


z={'h1':[1,2,2,3,4,5]}
z.keys(),z.values(),z.items()


num_attribs = list(housing)
cat_attribs = ["ocean_proximity"]


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


num_pipeline = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('std_scaler', StandardScaler()),
])


>>> housing_prepared = num_pipeline.fit_transform(X)


#housing
#housing.iloc[1:4, 2:4]
housing.iloc[:, :-1]


#housing_new = pd.concat([pd.DataFrame(housing_prepared),housing['ocean_proximity'].reset_index()],axis =1)
#housing_new = housing_new.drop({'index',8},axis=1)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,Y)


from sklearn.model_selection import train_test_split


pd.DataFrame(housing_prepared)


housing_prepared[:,8:9]


x_train,x_test,y_train,y_test=train_test_split(housing_prepared[:,:8],housing_prepared[:,8:9],test_size=0.20,random_state=400)


from sklearn.linear_model import LinearRegression


lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)


>>> from sklearn.metrics import mean_squared_error
>>> housing_predictions = lin_reg.predict(x_test)
>>> lin_mse = mean_squared_error(y_test, housing_predictions)
>>> lin_rmse = np.sqrt(lin_mse)
>>> lin_rmse


from sklearn.model_selection import StratifiedShuffleSplit


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


strat_train_set


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
X = imputer.transform(housing_num)
>>> imputer.statistics_ # to get meadian value array
>>> housing_num.median().values#manually checked value


>>> from sklearn.preprocessing import LabelEncoder
>>> encoder = LabelEncoder()
>>> housing_cat = housing["ocean_proximity"]
>>> housing_cat_encoded = encoder.fit_transform(housing_cat)
>>> housing_cat_encoded
print(encoder.classes_)


x_train_housing = 


from sklearn.tree import DecisionTreeRegressor


tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,Y)


>>> from sklearn.metrics import mean_squared_error
housing_predictions = tree_reg.predict(housing_prepared)
>>> tree_mse = mean_squared_error(Y, housing_predictions)
>>> tree_rmse = np.sqrt(tree_mse)
>>> tree_rmse


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared,Y,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


>>> def display_scores(scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)


from sklearn.model_selection import GridSearchCV


param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]


>>> from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared,Y)


grid_search.cv_results_


>>> cvres = grid_search.cv_results_
>>> for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


housing_num = housing.drop("ocean_proximity", axis=1)


num_attribs = list(housing_num)


feature_importances = grid_search.best_estimator_.feature_importances_
#extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#>>> cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs #+ extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


from sklearn import datasets, svm, metrics
digits = datasets.load_digits()


digits.keys()


x, y = digits["df"], digits["target"]


%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
some_digit = x[12]
some_digit_image = some_digit.reshape(8,8)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()


X_train, X_test, y_train, y_test = x[:1406], x[1406:], y[:1406], y[1406:]


import numpy as np
shuffle_index = np.random.permutation(1406)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


from sklearn.linear_model import SGDClassifier


y_train_5 = (y_train == 5) # doing to make binary classification


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


>>> sgd_clf.predict([some_digit])
#array([ True], dtype=bool)


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


>>> from sklearn.model_selection import cross_val_score
>>> cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


>>> from sklearn.metrics import confusion_matrix
>>> confusion_matrix(y_train_5, y_train_pred)


>>> from sklearn.metrics import precision_score, recall_score
>>> print(precision_score(y_train_5, y_train_pred))
>>> recall_score(y_train_5, y_train_pred)


>>> from sklearn.metrics import f1_score
>>> f1_score(y_train_5, y_train_pred)


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()


>>> from sklearn.metrics import roc_auc_score
>>> roc_auc_score(y_train_5, y_scores)


>>> sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
>>> sgd_clf.predict([some_digit])


>>> some_digit_scores = sgd_clf.decision_function([some_digit])
>>> some_digit_scores


#sgd_clf.classes_
print(sgd_clf.classes_)


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")


>>> forest_clf.fit(X_train, y_train)
>>> forest_clf.predict([some_digit])


forest_clf.predict_proba([some_digit]) #to get probability score for multiple category
#scikit learn defaultly detect the multiclass and starin the df


>>> from sklearn.preprocessing import StandardScaler
>>> scaler = StandardScaler()
>>> X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
>>> cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


>>> y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
>>> conf_mx = confusion_matrix(y_train, y_train_pred)
>>> conf_mx


from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


knn_clf.predict([some_digit])


>>> y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
>>> f1_score(y_train, y_train_knn_pred, average="macro")


noise = np.random.randint(0, 100, (len(X_train), 64))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 64))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


from sklearn.linear_model import SGDRegressor
#sgd_reg = SGDRegressor(max_iter =50, penalty=None, eta0=0.1)
#sgd_reg.fit(X_train, y_train_5)


>>> sgd_reg.intercept_, sgd_reg.coef_


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)


>>> from sklearn.preprocessing import PolynomialFeatures
>>> poly_features = PolynomialFeatures(degree=2, include_bias=False)
>>> X_poly = poly_features.fit_transform(X)
X[0]


>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X_poly, y)
>>> lin_reg.intercept_, lin_reg.coef_


#cross validation is best way to find ubderfitting and overfitting of model


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")


lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)


from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((
("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
("lin_reg", LinearRegression()),
))
plot_learning_curves(polynomial_regression, X, y)


#One way to improve an overfitting model is to feed it more training df until the validation error reaches the training error.


>>> from sklearn.linear_model import Ridge
>>> ridge_reg = Ridge(alpha=1, solver="cholesky")
>>> ridge_reg.fit(X, y)
>>> ridge_reg.predict([[1.5]])


>>> sgd_reg = SGDRegressor(penalty="l2")
>>> sgd_reg.fit(X, y.ravel())
>>> sgd_reg.predict([[1.5]])


>>> from sklearn.linear_model import Lasso
>>> lasso_reg = Lasso(alpha=0.1)
>>> lasso_reg.fit(X, y)
>>> lasso_reg.predict([[1.5]])


#elastic net is middle ground r=0 ridge,r=1 Lasso


>>> from sklearn.linear_model import ElasticNet
>>> elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
>>> elastic_net.fit(X, y)
>>> elastic_net.predict([[1.5]])


from sklearn.base import clone
sgd_reg = SGDRegressor(max_iter =1, warm_start=True, penalty=None,learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)


>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> list(iris.keys())
>>> X = iris["df"][:, 3:] # petal width
>>> y = (iris["target"] == 2).astype(np.int)


from sklearn.linear_model import LogisticRegression
X = iris["df"][:, (2, 3)] # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)


>>> softmax_reg.predict([[5, 2]])
>>> softmax_reg.predict_proba([[5, 2]])


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X = iris["df"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
svm_clf = Pipeline((
("scaler", StandardScaler()),
("linear_svc", LinearSVC(C=1, loss="hinge")),
))
svm_clf.fit(X, y)


svm_clf.predict([[5.5, 1.7]])


from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline((
("poly_features", PolynomialFeatures(degree=3)),("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
))
polynomial_svm_clf.fit(X, y)


from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))
poly_kernel_svm_clf.fit(X, y)


if
your model is overfitting, you might want to reduce the polynomial degree. Conversely, if it is
underfitting, you can try increasing it. The hyperparameter coef0 controls how much the model is
influenced by high-degree polynomials versus low-degree polynomials.


rbf_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
))
rbf_kernel_svm_clf.fit(X, y)


Increasing gamma makes the bell-shape curve
narrower (see the left plot of Figure 5-8), and as a result each instance’s range of influence is smaller: the
decision boundary ends up being more irregular, wiggling around individual instances. Conversely, a
small gamma value makes the bell-shaped curve wider, so instances have a larger range of influence, and
the decision boundary ends up smoother. So γ acts like a regularization hyperparameter: if your model is
overfitting, you should reduce it, and if it is underfitting, you should increase it (similar to the C
hyperparameter).