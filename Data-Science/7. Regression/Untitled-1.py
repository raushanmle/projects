
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing Library
import pandas as pd
from pathlib import Path

df = pd.read_csv(str(Path().resolve().parent) +
                 "\\4. DataFrame\\sample-data\\kc_housingdata.csv")

df1 = df.drop('date', axis=1)
df1 = pd.get_dummies(df1)

x = df1
y = df1['price']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=400)

reg = LinearRegression()
reg.fit(x_train.drop(['id', 'zipcode', 'price'], axis=1), y_train)
reg.score(x_test.drop(['id', 'zipcode', 'price'], axis=1), y_test)

reg.coef_
reg.intercept_
reg.singular_

x_data = x_test.drop(['id', 'zipcode'], axis=1)
x_predicted_data = pd.DataFrame(reg.predict(
    x_test.drop(['id', 'zipcode', 'price'], axis=1)), index=None)
x_data = x_data.reset_index()
x_test['price'].reset_index()

pd.concat([x_data, x_predicted_data, x_test['price'].reset_index()], axis=1)
a1 = x_test.drop(['id', 'zipcode', 'price'], axis=1).iloc[[0, 1], :]
reg.predict(a1)

df3 = pd.read_csv("C:\\Users\\Raushan Kumar\\Downloads\IRIS.csv", header=None)


df3['result'] = np.where(df3[8] == 'setosa', 1, 0)
