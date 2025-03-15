import matplotlib
import statsmodels.tsa.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt # Visualization
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima
from colorama import Fore
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

############# Data Analysis ##############
df = pd.read_csv("AirPassengers.csv")
df.head()
df.plot()

ss_decomposition = seasonal_decompose(x=df['#Passengers'].values, model='additive', period=12)
estimated_trend = ss_decomposition.trend
estimated_seasonal = ss_decomposition.seasonal
estimated_residual = ss_decomposition.resid
plt.plot(estimated_trend)
plt.plot(estimated_seasonal)
plt.plot(estimated_residual)
plt.title('Air Passengers detrended by subtracting the trend component', fontsize=16)
plt.show()

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

# Draw Plot
plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':120})
autocorrelation_plot(df['#Passengers'].tolist())

fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df['#Passengers'].tolist(), lags=50, ax=axes[0])
plot_pacf(df['#Passengers'].tolist(), lags=50, ax=axes[1])

# For non-seasonal data
#p=1, d=1, q=0 or 1
model=ARIMA(df['#Passengers'],order=(1,0,1))
model_fit=model.fit()
model_fit.summary()
df['forecast']=model_fit.predict(start=100,end=143,dynamic=True)
df[['#Passengers','forecast']].plot(figsize=(12,8))

model=sm.SARIMAX(df['#Passengers'],order=(1, 0, 1),seasonal_order=(1,0,1,12))
results=model.fit()


df['forecast1']=results.predict(start=100,end=143,dynamic=True)
df[['#Passengers','forecast1']].plot(figsize=(12,8))

df['shifted_12_months'] = df['#Passengers'].diff(periods=12)

df['ds'] = pd.to_datetime(df['Month'], format = '%Y-%m')
df['y'] = df['#Passengers']

###############based on TF ############
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore

# Prepare the data
df['ds'] = pd.to_datetime(df['Month'], format='%Y-%m')
df['y'] = df['#Passengers']

# Split the dataset into train and test set
train_size = int(0.85 * len(df))
train = df.iloc[:train_size, :]
test = df.iloc[train_size:, :]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train[['y']])
test_scaled = scaler.transform(test[['y']])

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

look_back = 1
X_train, y_train = create_dataset(train_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model summary
model.summary()
# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=20, validation_data=(X_test, y_test), verbose=2)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Get the root mean squared error (RMSE) and MAE
score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
score_mae = mean_absolute_error(y_test[0], test_predict[:, 0])
print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))
print(Fore.GREEN + 'MAE: {}'.format(score_mae))

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(df['ds'], df['y'], label='Actual')
plt.plot(df['ds'].iloc[look_back:train_size], train_predict, label='Train Predict')
plt.plot(df['ds'].iloc[train_size + look_back:], test_predict, label='Test Predict')
plt.legend()
plt.show()