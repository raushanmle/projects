import matplotlib
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.tsa.stattools as sm
import seaborn as sns # Visualization
import matplotlib.pyplot as plt # Visualization
%matplotlib inline
from colorama import Fore
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

import warnings # Supress warnings 
warnings.filterwarnings('ignore')

path = "C:\\Users\\Raushan\\Downloads\\Code\\time series\\"
df = pd.read_csv(path + "AirPassengers.csv")

df.head()

df.plot()



from statsmodels.tsa.seasonal import seasonal_decompose
ss_decomposition = seasonal_decompose(x=df['#Passengers'].values, model='additive', freq=12)
estimated_trend = ss_decomposition.trend
estimated_seasonal = ss_decomposition.seasonal
estimated_residual = ss_decomposition.resid
estimated_trend.plot()
ss_decomposition[0].plot()

plt.plot(estimated_trend)
plt.plot(estimated_seasonal)
plt.plot(estimated_residual)
plt.show()

plt.title('Air Passengers detrended by subtracting the trend component', fontsize=16)

import pmdarima

results = pmdarima.arima.CHTest(m=12).estimate_seasonal_differencing_term(df['#Passengers'])

test = pmdarima.arima.PPTest() # You can choose alpha here, default = 0.05
test.should_diff(df['#Passengers'])


#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=sm.adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(df['#Passengers'])

#P value greater than 0.05 non stationary

# Test for seasonality
from pandas.plotting import autocorrelation_plot

# Draw Plot
plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':120})
autocorrelation_plot(df['#Passengers'].tolist())

fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df['#Passengers'].tolist(), lags=50, ax=axes[0])
plot_pacf(df['#Passengers'].tolist(), lags=50, ax=axes[1])


# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(df['#Passengers'],order=(1,0,1))
model_fit=model.fit()

model_fit.summary()

df['forecast']=model_fit.predict(start=100,end=143,dynamic=True)
df[['#Passengers','forecast']].plot(figsize=(12,8))

import statsmodels.api as sm

model=sm.tsa.statespace.SARIMAX(df['#Passengers'],order=(1, 0, 1),seasonal_order=(1,0,1,12))
results=model.fit()


df['forecast1']=results.predict(start=100,end=143,dynamic=True)
df[['#Passengers','forecast1']].plot(figsize=(12,8))

df['shifted_12_months'] = df['#Passengers'].diff(periods=12)

df['ds'] = pd.to_datetime(df['Month'], format = '%Y-%m')
df['y'] = df['#Passengers']
from fbprophet import Prophet


# Train the model
model = Prophet()
#model.add_regressor('rainfall')
#model.add_regressor('temperature')
#model.add_regressor('drainage_volume')
#model.add_regressor('river_hydrometry')

# Fit the model with train set
model.fit(df[['ds','y']])

# Predict on valid set
y_pred = model.predict(x_valid)

# Calcuate metrics
score_mae = mean_absolute_error(y_valid, y_pred['yhat'])
score_rmse = math.sqrt(mean_squared_error(y_valid, y_pred['yhat']))

print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))

from keras.models import Sequential
from keras.layers import Dense, LSTM

#Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

train_size = int(0.85 * len(df))
train = df.iloc[:train_size, :]
x_train, y_train = pd.DataFrame(df.iloc[:train_size,[0]]), pd.DataFrame(df.iloc[:train_size, 1])
x_valid, y_valid = pd.DataFrame(df.iloc[train_size:, [0,2,3,4,5]]), pd.DataFrame(df.iloc[train_size:, 1])
dataset = df.set_index('Month')

# Split the dataset into train and test set
dataset = dataset.values
train_size = int(dataset.shape[0] * 0.67)
train_df, test_df = dataset[:train_size, :], dataset[train_size:, :]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_df_rescaled = scaler.fit_transform(train_df)
test_df_rescaled = scaler.transform(test_df)

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    m = len(dataset)
    X = []
    y = []
    for i in range(look_back, m):
        X.append(dataset[i - look_back: i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)


look_back = 1
X_train_rescaled, y_train_rescaled = create_dataset(train_df_rescaled, look_back=look_back)
X_test_rescaled, y_test_rescaled = create_dataset(test_df_rescaled, look_back=look_back)

X_train = train_df.reshape(-1, train_df.shape[0],1)
X_train = np.delete(X_train, 0, axis=1)

Y_train = y_train_rescaled.reshape(-1, y_train_rescaled.shape[0],1)
#X_train = df['#Passengers'].values.reshape((1, df.shape[0], 1))


model = Sequential()
model.add(LSTM(4, input_shape=(95, 1),return_sequences=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, np.array([[y_train_rescaled]]), epochs=20, batch_size=1, verbose=2)


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5, validation_data=(x_test, y_test))

model.summary()




# Lets predict with the model
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])

test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Get the root mean squared error (RMSE) and MAE
score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
score_mae = mean_absolute_error(y_test[0], test_predict[:,0])
print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))

from numpy import array
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
data = data.reshape((1, 10, 1))
print(data.shape)
