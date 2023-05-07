# Author : Raushan Kumar
# Objective of this code : You'll learn working flow of forecsting models

# Importing libraries
import matplotlib.pyplot as plt
from sklearn import metrics
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
import hurst
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from scipy.stats.stats import pearsonr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from urllib.request import urlopen
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.width", 100)
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.api import anova_lm
%matplotlib inline


df = pd.read_excel(".\\AirQualityUCI.xlsx")

print (
'Mean: ' , np. mean( df[ 'T' ]), 
'Standard Deviation: ' , np. std( df[ 'T' ]),
'Maximum Temperature: ' , max( df[ 'T' ]),'Minimum Temperature: ' , min( df[ 'T' ]))

df['T_t-1'] = df['T'].shift(1)
df_naive = df[['T','T_t-1']][1:]
true = df_naive['T']
prediction = df_naive['T_t-1']
error = sqrt(metrics.mean_squared_error(true,prediction))
print ('RMSE for Naive Method 1: ', error)

#calculate rolling sum then mean for those and then shifted to 1 unit down
df['T_rm'] = df['T'].rolling(3).mean().shift(1)
df_naive = df[['T','T_rm']].dropna()
true = df_naive['T']
prediction = df_naive['T_rm']
error = sqrt(metrics.mean_squared_error(true,prediction))
print ('RMSE for Naive Method 2: ', error)

split = len(df) - int(0.2*len(df))
train, test = df['T'][0:split], df['T'][split:]

#AIC and BIC should be lower for a best model
plot_acf(train, lags = 100)
plt.show()

plot_pacf(train, lags = 100)
plt.show()

# #### ADF test
result = adfuller(train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

sunspots = sm.datasets.sunspots.load_pandas().data
sunspots.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del sunspots["YEAR"]
sunspots.plot(figsize=(12,8))

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

adf_test(sunspots['SUNACTIVITY']) #data non stationary as p values>.05
kpss_test(sunspots['SUNACTIVITY'])


# Based upon the significance level of 0.05 and the p-value of the KPSS test, the null hypothesis can not be rejected. Hence, the series is stationary as per the KPSS test.

# It is always better to apply both the tests, so that it can be ensured that the series is truly stationary. Possible outcomes of applying these stationary tests are as follows:

# Case 1: Both tests conclude that the series is not stationary - The series is not stationary
# Case 2: Both tests conclude that the series is stationary - The series is stationary
# Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
# Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
# Here, due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary. The series can be detrended by differencing or by model fitting.


sunspots['SUNACTIVITY_diff'] = sunspots['SUNACTIVITY'] - sunspots['SUNACTIVITY'].shift(1)
sunspots['SUNACTIVITY_diff'].dropna().plot(figsize=(12,8))

adf_test(sunspots['SUNACTIVITY_diff'].dropna())
kpss_test(sunspots['SUNACTIVITY_diff'].dropna())

# Hurst to test stationary
H, c,data = hurst.compute_Hc(train)
print("H = {:.4f}, c = {:.4f}".format(H,c))

# The value of H<0.5 shows anti-persistent behavior, and H>0.5 shows persistent behavior or a trending
# series. H=0.5 shows random
# walk/Brownian motion. The value of H<0.5, confirming that our series is
# stationary.

model = ARIMA(train.values, order=(5, 0, 2))
model_fit = model.fit()

predictions = model_fit.predict(len(test))
test_ = pd.DataFrame(test)
test_['predictions'] = predictions[0:1871]

plt.plot(df['T'])
plt.plot(test_.predictions)
plt.show()

error =sqrt(metrics.mean_squared_error(test.values,predictions[0:1871]))
print ('Test RMSE for ARIMA: ', error)

df_multi = df[['T', 'C6H6(GT)']]
split = len(df) - int(0.2*len(df))
train_multi, test_multi = df_multi[0:split], df_multi[split:]

model = VARMAX(train_multi, order = (2,1))
model_fit = model.fit()
predictions_multi = model_fit.forecast( steps=len(test_multi))

plt.plot(train_multi['T'])
plt.plot(test_multi['T'])
plt.plot(predictions_multi.iloc[:,0:1], '--')
plt.show()
plt.plot(train_multi['C6H6(GT)'])
plt.plot(test_multi['C6H6(GT)'])
plt.plot(predictions_multi.iloc[:,1:2], '--')
plt.show()

x = train_multi['T'].values
y = train_multi['C6H6(GT)'].values
corr , p = pearsonr(x,y)
print("Corelation Coefficient = "+ str(corr),'\nP-Value =',p)

model = SARIMAX(x, exog = y, order = (2, 0, 2), seasonal_order = (2, 0, 1,3), enforce_stationarity=False, enforce_invertibility = False)
model_fit = model.fit()

y_ = test_multi['C6H6(GT)'].values
predicted = model_fit.predict(exog=y_)
test_multi_ = pd.DataFrame(test)
test_multi_['predictions'] = predicted[0:1871]
plt.plot(train_multi['T'])
plt.plot(test_multi_['T'])
plt.plot(test_multi_.predictions, '--')

model = ExponentialSmoothing(train.values, trend='add')
model_fit = model.fit()

predictions_ = model_fit.predict(len(test))
plt.plot(test.values)
plt.plot(predictions_[1:1871])

prediction = []
data = train.values
for t in test.values:
    model = (ExponentialSmoothing(data).fit())
    y = model.predict()
    prediction.append(y[0])
    data = np.append(data, t)

test_ = pd.DataFrame(test)
test_['predictionswf'] = prediction

plt.plot(test_['T'])
plt.plot(test_.predictionswf, '--')
plt.show()

error = sqrt(metrics.mean_squared_error(test.values,prediction))
print ('Test RMSE for Triple Exponential Smoothing with Walk-Forward Validation: ', error)

x = np.arange (1,500,1)
y = 0.4 * x + 30
plt.plot(x,y)

trainx, testx = x[0:int(0.8*(len(x)))], x[int(0.8*(len(x))):]
trainy, testy = y[0:int(0.8*(len(y)))], y[int(0.8*(len(y))):]
train = np.array(list(zip(trainx,trainy)))
test = np.array(list(zip(trainx,trainy)))

def create_dataset(n_X, look_back):
    dataX, dataY = [], []
    for i in range(len(n_X)-look_back):
        a = n_X[i:(i+look_back), ]
        dataX.append(a)
        dataY.append(n_X[i + look_back, ])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainx,trainy = create_dataset(train, look_back)
testx,testy = create_dataset(test, look_back)
trainx = np.reshape(trainx, (trainx.shape[0], 1, 2))
testx = np.reshape(testx, (testx.shape[0], 1, 2))

model = Sequential()
model.add(LSTM(256, return_sequences = True, input_shape =(trainx.shape[1], 2)))
model.add(LSTM(128,input_shape = (trainx.shape[1], 2)))
model.add(Dense(2))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(trainx, trainy, epochs = 2000, batch_size = 10, verbose = 2,
shuffle = False)
# model.save_weights('LSTMBasic1.h5')
# model.load_weights('LSTMBasic1.h5')
predict = model.predict(testx)

plt.plot(testx.reshape(398,2)[:,0:1], testx.reshape(398,2)[:,1:2])
plt.plot(predict[:,0:1], predict[:,1:2])

x = np.arange (1,500,1)
y = np.sin(x)
plt.plot(x,y)

trainx, testx = x[0:int(0.8*(len(x)))], x[int(0.8*(len(x))):]
trainy, testy = y[0:int(0.8*(len(y)))], y[int(0.8*(len(y))):]
train = np.array(list(zip(trainx,trainy)))
test = np.array(list(zip(trainx,trainy)))

look_back = 1
trainx,trainy = create_dataset(train, look_back)
testx,testy = create_dataset(test, look_back)
trainx = np.reshape(trainx, (trainx.shape[0], 1, 2))
testx = np.reshape(testx, (testx.shape[0], 1, 2))

model = Sequential()
model.add(LSTM(512, return_sequences = True, input_shape =
(trainx.shape[1], 2)))
model.add(LSTM(256,input_shape = (trainx.shape[1], 2)))
model.add(Dense(2))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(trainx, trainy, epochs = 2000, batch_size = 10, verbose = 2,
shuffle = False)
# model.save_weights('LSTMBasic2.h5')
# model.load_weights('LSTMBasic2.h5')
predict = model.predict(testx)

plt.plot(trainx.reshape(398,2)[:,0:1], trainx.reshape(398,2)[:,1:2])
plt.plot(predict[:,0:1], predict[:,1:2])

# # Examples
np.random.seed(9876789)
nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x-5)**2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

res = sm.OLS(y, X).fit()
print(res.summary())

print('Parameters: ', res.params)
print('Standard errors: ', res.bse)
print('Predicted values: ', res.predict())

prstd, iv_l, iv_u = wls_prediction_std(res)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="data")
ax.plot(x, y_true, 'b-', label="True")
ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');

url = 'http://stats191.stanford.edu/data/salary.table'
fh = urlopen(url)
salary_table = pd.read_table(fh)
# salary_table.to_csv('salary.table')

plt.figure(figsize=(6,6))
symbols = ['D', '^']
colors = ['r', 'g', 'blue']
factor_groups = salary_table.groupby(['E','M'])
for values, group in factor_groups:
    i,j = values
    plt.scatter(group['X'], group['S'], marker=symbols[j], color=colors[i-1],
               s=100)
plt.xlabel('Experience');
plt.ylabel('Salary');

formula = 'S ~ C(E) + C(M) + X'
lm = ols(formula, salary_table).fit()
print(lm.summary())

lm.model.exog
infl = lm.get_influence()
print(infl.summary_table())

df_infl = infl.summary_frame()
resid = lm.resid
plt.figure(figsize=(6,6));
for values, group in factor_groups:
    i,j = values
    group_num = i*2 + j - 1  # for plotting purposes
    x = [group_num] * len(group)
    plt.scatter(x, resid[group.index], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
plt.xlabel('Group');
plt.ylabel('Residuals');

interX_lm = ols("S ~ C(E) * X + C(M)", salary_table).fit()
print(interX_lm.summary())

table1 = anova_lm(lm, interX_lm)
print(table1)

interM_lm = ols("S ~ X + C(E)*C(M)", data=salary_table).fit()
print(interM_lm.summary())

table2 = anova_lm(lm, interM_lm)
print(table2)


pip install pandas_datareader

 [markdown]
# # Time Series


%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, pacf, graphics


sns.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc('figure',figsize=(16, 6))


data = pdr.get_data_fred('HOUSTNSA', '1959-01-01', '2019-06-01')
housing = data.HOUSTNSA.pct_change().dropna()
# Scale by 100 to get percentages
housing = 100 * housing.asfreq('MS')
fig, ax = plt.subplots()
ax = housing.plot(ax=ax)


mod = AutoReg(housing, 3)
res = mod.fit()
print(res.summary())


res = mod.fit(cov_type="HC0")
print(res.summary())


sel = ar_select_order(housing, 13)
sel.ar_lags


res = sel.model.fit()
print(res.summary())


fig = res.plot_predict(720, 840)


fig = plt.figure(figsize=(16,9))
fig = res.plot_diagnostics(fig=fig, lags=30)


sel = ar_select_order(housing, 13, seasonal=True)
sel.ar_lags
res = sel.model.fit()
print(res.summary())


fig = res.plot_predict(720, 840)


fig = plt.figure(figsize=(16,9))
fig = res.plot_diagnostics(fig=fig, lags=30)


yoy_housing = data.HOUSTNSA.pct_change(12).resample("MS").last().dropna()
_, ax = plt.subplots()
ax = yoy_housing.plot(ax=ax)


sel = ar_select_order(yoy_housing, 13)
sel.ar_lags


sel = ar_select_order(yoy_housing, 13, glob=True)
sel.ar_lags
res = sel.model.fit()
print(res.summary())


sel = ar_select_order(yoy_housing, 13, glob=True, seasonal=True)
sel.ar_lags
res = sel.model.fit()
print(res.summary())


fig = plt.figure(figsize=(16,9))
fig = res.plot_diagnostics(fig=fig, lags=30)


sel = ar_select_order(yoy_housing, 13, glob=True, seasonal=True, old_names=False)
sel.ar_lags
res = sel.model.fit()
print(res.summary())


data = pdr.get_data_fred('INDPRO', '1959-01-01', '2019-06-01')
ind_prod = data.INDPRO.pct_change(12).dropna().asfreq('MS')
_, ax = plt.subplots(figsize=(16,9))
ind_prod.plot(ax=ax)


sel = ar_select_order(ind_prod, 13, 'bic')
res = sel.model.fit()
print(res.summary())


sel = ar_select_order(ind_prod, 13, 'bic', glob=True)
sel.ar_lags
res_glob = sel.model.fit()
print(res.summary())


ind_prod.shape


fig = res_glob.plot_predict(start=714, end=732)


res_ar5 = AutoReg(ind_prod, 5).fit()
predictions = pd.DataFrame({"AR(5)": res_ar5.predict(start=714, end=726),
                            "AR(13)": res.predict(start=714, end=726),
                            "Restr. AR(13)": res_glob.predict(start=714, end=726)})
_, ax = plt.subplots()
ax = predictions.plot(ax=ax)


fig = plt.figure(figsize=(16,9))
fig = res_glob.plot_diagnostics(fig=fig, lags=30)


import np as np
start = ind_prod.index[-24]
forecast_index = pd.date_range(start, freq=ind_prod.index.freq, periods=36)
cols = ['-'.join(str(val) for val in (idx.year, idx.month)) for idx in forecast_index]
forecasts = pd.DataFrame(index=forecast_index,columns=cols)
for i in range(1, 24):
    fcast = res_glob.predict(start=forecast_index[i], end=forecast_index[i+12], dynamic=True)
    forecasts.loc[fcast.index, cols[i]] = fcast
_, ax = plt.subplots(figsize=(16, 10))
ind_prod.iloc[-24:].plot(ax=ax, color="black", linestyle="--")
ax = forecasts.plot(ax=ax)


from statsmodels.tsa.api import SARIMAX

sarimax_mod = SARIMAX(ind_prod, order=((1,5,12,13),0, 0), trend='c')
sarimax_res = sarimax_mod.fit()
print(sarimax_res.summary())


sarimax_params = sarimax_res.params.iloc[:-1].copy()
sarimax_params.index = res_glob.params.index
params = pd.concat([res_glob.params, sarimax_params], axis=1, sort=False)
params.columns = ["AutoReg", "SARIMAX"]
params


from statsmodels.tsa.deterministic import DeterministicProcess


dp = DeterministicProcess(housing.index, constant=True, period=12, fourier=2)
mod = AutoReg(housing,2, trend="n",seasonal=False, deterministic=dp)
res = mod.fit()
print(res.summary())


pip install statsmodels


import np as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


from statsmodels.graphics.api import qqplot


print(sm.datasets.sunspots.NOTE)


dta = sm.datasets.sunspots.load_pandas().data


dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]


dta.plot(figsize=(12,8));


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)


arma_mod20 = ARIMA(dta, order=(2, 0, 0)).fit()


print(arma_mod20.params)


arma_mod30 = ARIMA(dta, order=(3, 0, 0)).fit()


print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)


print(arma_mod30.params)


print(arma_mod30.aic,arma_mod30.bic, arma_mod30.hqic)


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax);


resid = arma_mod30.resid


stats.normaltest(resid)


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)


sm.tsa.acf(resid.values.squeeze(), fft=True, qstat=True)


np.c_[range(1,41), r[1:], q, p]


r,q,p = sm.tsa.acf(resid.values.squeeze(), fft=True, qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


predict_sunspots = arma_mod30.predict('1990', '2012', dynamic=True)
print(predict_sunspots)


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


mean_forecast_err(dta.SUNACTIVITY, predict_sunspots)


dta = sm.datasets.macrodata.load_pandas().data


index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
print(index)


dta.index = index
del dta['year']
del dta['quarter']


print(sm.datasets.macrodata.NOTE)


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
dta.realgdp.plot(ax=ax);
legend = ax.legend(loc = 'upper left');
legend.prop.set_size(20)


gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(dta.realgdp)


gdp_decomp = dta[['realgdp']].copy()
gdp_decomp["cycle"] = gdp_cycle
gdp_decomp["trend"] = gdp_trend


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
gdp_decomp[["realgdp", "trend"]]["2000-03-31":].plot(ax=ax, fontsize=16);
legend = ax.get_legend()
legend.prop.set_size(20);


bk_cycles = sm.tsa.filters.bkfilter(dta[["infl","unemp"]])


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
bk_cycles.plot(ax=ax, style=['r--', 'b-'])


print(sm.tsa.stattools.adfuller(dta['unemp'])[:3])


print(sm.tsa.stattools.adfuller(dta['infl'])[:3])


cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl","unemp"]])
print(cf_cycles.head(10))


fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
cf_cycles.plot(ax=ax, style=['r--','b-'])


dta[["infl","unemp"]]

 [markdown]
# # Exponential smoothing


import os
import np as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
%matplotlib inline


data = [446.6565,  454.4733,  455.663 ,  423.6322,  456.2713,  440.5881, 425.3325,  485.1494,  506.0482,  526.792 ,  514.2689,  494.211 ]
index= pd.date_range(start='1996', end='2008', freq='A')
oildata = pd.Series(data, index)

data = [17.5534,  21.86  ,  23.8866,  26.9293,  26.8885,  28.8314, 30.0751,  30.9535,  30.1857,  31.5797,  32.5776,  33.4774, 39.0216,  41.3864,  41.5966]
index= pd.date_range(start='1990', end='2005', freq='A')
air = pd.Series(data, index)

data = [263.9177,  268.3072,  260.6626,  266.6394,  277.5158,  283.834 , 290.309 ,  292.4742,  300.8307,  309.2867,  318.3311,  329.3724, 338.884 ,  339.2441,  328.6006,  314.2554,  314.4597,  321.4138, 329.7893,  346.3852,  352.2979,  348.3705,  417.5629,  417.1236, 417.7495,  412.2339,  411.9468,  394.6971,  401.4993,  408.2705, 414.2428]
index= pd.date_range(start='1970', end='2001', freq='A')
livestock2 = pd.Series(data, index)

data = [407.9979 ,  403.4608,  413.8249,  428.105 ,  445.3387,  452.9942, 455.7402]
index= pd.date_range(start='2001', end='2008', freq='A')
livestock3 = pd.Series(data, index)

data = [41.7275,  24.0418,  32.3281,  37.3287,  46.2132,  29.3463, 36.4829,  42.9777,  48.9015,  31.1802,  37.7179,  40.4202, 51.2069,  31.8872,  40.9783,  43.7725,  55.5586,  33.8509, 42.0764,  45.6423,  59.7668,  35.1919,  44.3197,  47.9137]
index= pd.date_range(start='2005', end='2010-Q4', freq='QS-OCT')
aust = pd.Series(data, index)


ax=oildata.plot()
ax.set_xlabel("Year")
ax.set_ylabel("Oil (millions of tonnes)")
print("Figure 7.1: Oil production in Saudi Arabia from 1996 to 2007.")


import os
import np as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
%matplotlib inline


fit1 = SimpleExpSmoothing(oildata, initialization_method="heuristic").fit(smoothing_level=0.2,optimized=False)
fcast1 = fit1.forecast(3).rename(r'$\alpha=0.2$')
fit2 = SimpleExpSmoothing(oildata, initialization_method="heuristic").fit(smoothing_level=0.6,optimized=False)
fcast2 = fit2.forecast(3).rename(r'$\alpha=0.6$')
fit3 = SimpleExpSmoothing(oildata, initialization_method="estimated").fit()
fcast3 = fit3.forecast(3).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])

plt.figure(figsize=(12, 8))
plt.plot(oildata, marker='o', color='black')
plt.plot(fit1.fittedvalues, marker='o', color='blue')
line1, = plt.plot(fcast1, marker='o', color='blue')
plt.plot(fit2.fittedvalues, marker='o', color='red')
line2, = plt.plot(fcast2, marker='o', color='red')
plt.plot(fit3.fittedvalues, marker='o', color='green')
line3, = plt.plot(fcast3, marker='o', color='green')
plt.legend([line1, line2, line3], [fcast1.name, fcast2.name, fcast3.name])


fit1 = Holt(air, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
fcast1 = fit1.forecast(5).rename("Holt's linear trend")
fit2 = Holt(air, exponential=True, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
fcast2 = fit2.forecast(5).rename("Exponential trend")
fit3 = Holt(air, damped_trend=True, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2)
fcast3 = fit3.forecast(5).rename("Additive damped trend")

plt.figure(figsize=(12, 8))
plt.plot(air, marker='o', color='black')
plt.plot(fit1.fittedvalues, color='blue')
line1, = plt.plot(fcast1, marker='o', color='blue')
plt.plot(fit2.fittedvalues, color='red')
line2, = plt.plot(fcast2, marker='o', color='red')
plt.plot(fit3.fittedvalues, color='green')
line3, = plt.plot(fcast3, marker='o', color='green')
plt.legend([line1, line2, line3], [fcast1.name, fcast2.name, fcast3.name])


fit1 = SimpleExpSmoothing(livestock2, initialization_method="estimated").fit()
fit2 = Holt(livestock2, initialization_method="estimated").fit()
fit3 = Holt(livestock2,exponential=True, initialization_method="estimated").fit()
fit4 = Holt(livestock2,damped_trend=True, initialization_method="estimated").fit(damping_trend=0.98)
fit5 = Holt(livestock2,exponential=True, damped_trend=True, initialization_method="estimated").fit()
params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$l_0$","$b_0$","SSE"] ,columns=['SES', "Holt's","Exponential", "Additive", "Multiplicative"])
results["SES"] =            [fit1.params[p] for p in params] + [fit1.sse]
results["Holt's"] =         [fit2.params[p] for p in params] + [fit2.sse]
results["Exponential"] =    [fit3.params[p] for p in params] + [fit3.sse]
results["Additive"] =       [fit4.params[p] for p in params] + [fit4.sse]
results["Multiplicative"] = [fit5.params[p] for p in params] + [fit5.sse]
results


for fit in [fit2,fit4]:
    pd.DataFrame(np.c_[fit.level,fit.trend]).rename(
        columns={0:'level',1:'slope'}).plot(subplots=True)
plt.show()
print('Figure 7.4: Level and slope components for Holt’s linear trend method and the additive damped trend method.')


fit1 = SimpleExpSmoothing(livestock2, initialization_method="estimated").fit()
fcast1 = fit1.forecast(9).rename("SES")
fit2 = Holt(livestock2, initialization_method="estimated").fit()
fcast2 = fit2.forecast(9).rename("Holt's")
fit3 = Holt(livestock2, exponential=True, initialization_method="estimated").fit()
fcast3 = fit3.forecast(9).rename("Exponential")
fit4 = Holt(livestock2, damped_trend=True, initialization_method="estimated").fit(damping_trend=0.98)
fcast4 = fit4.forecast(9).rename("Additive Damped")
fit5 = Holt(livestock2, exponential=True, damped_trend=True, initialization_method="estimated").fit()
fcast5 = fit5.forecast(9).rename("Multiplicative Damped")

ax = livestock2.plot(color="black", marker="o", figsize=(12,8))
livestock3.plot(ax=ax, color="black", marker="o", legend=False)
fcast1.plot(ax=ax, color='red', legend=True)
fcast2.plot(ax=ax, color='green', legend=True)
fcast3.plot(ax=ax, color='blue', legend=True)
fcast4.plot(ax=ax, color='cyan', legend=True)
fcast5.plot(ax=ax, color='magenta', legend=True)
ax.set_ylabel('Livestock, sheep in Asia (millions)')
plt.show()
print('Figure 7.5: Forecasting livestock, sheep in Asia: comparing forecasting performance of non-seasonal methods.')


fit1 = ExponentialSmoothing(aust, seasonal_periods=4, trend='add', seasonal='add', use_boxcox=True, initialization_method="estimated").fit()
fit2 = ExponentialSmoothing(aust, seasonal_periods=4, trend='add', seasonal='mul', use_boxcox=True, initialization_method="estimated").fit()
fit3 = ExponentialSmoothing(aust, seasonal_periods=4, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
fit4 = ExponentialSmoothing(aust, seasonal_periods=4, trend='add', seasonal='mul', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
results=pd.DataFrame(index=[r"$\alpha$",r"$\beta$",r"$\phi$",r"$\gamma$",r"$l_0$","$b_0$","SSE"])
params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
results["Additive"]       = [fit1.params[p] for p in params] + [fit1.sse]
results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
results["Additive Dam"]   = [fit3.params[p] for p in params] + [fit3.sse]
results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]

ax = aust.plot(figsize=(10,6), marker='o', color='black', title="Forecasts from Holt-Winters' multiplicative method" )
ax.set_ylabel("International visitor night in Australia (millions)")
ax.set_xlabel("Year")
fit1.fittedvalues.plot(ax=ax, style='--', color='red')
fit2.fittedvalues.plot(ax=ax, style='--', color='green')

fit1.forecast(8).rename('Holt-Winters (add-add-seasonal)').plot(ax=ax, style='--', marker='o', color='red', legend=True)
fit2.forecast(8).rename('Holt-Winters (add-mul-seasonal)').plot(ax=ax, style='--', marker='o', color='green', legend=True)

plt.show()
print("Figure 7.6: Forecasting international visitor nights in Australia using Holt-Winters method with both additive and multiplicative seasonality.")

results


