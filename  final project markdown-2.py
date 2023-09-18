#!/usr/bin/env python
# coding: utf-8

# # Time series analysis of Tesla

# In[26]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statsmodels.api as sm
import arch
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[19]:


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path_Tesla = os.path.join(desktop_path, "/Users/apple/Desktop/semester 3/FBE543/Tesladata.xlsx")
data_Tesla=pd.read_excel(file_path_Tesla,index_col='Date',parse_dates=True)


# ## Graph the daily return 

# In[11]:


y=data_Tesla["Daily return"]
plt.figure(figsize=(12, 4))
plt.plot(y,color="b")
plt.title("Tesla daily return")
plt.xlabel("Date")
plt.ylabel("Daily return")


# ## 1st difference

# In[13]:


d1return = np.diff(y)
plt.figure(figsize=(12, 4))
plt.plot(d1return)
plt.title('First-order Difference of daily return')
plt.xlabel('Date')
plt.ylabel('Differnece of return')


# ## 2nd difference

# In[196]:


d2return = np.diff(y, n=2)
plt.figure(figsize=(12, 4))
plt.plot(x[2:],d2return)
plt.title('Sceond-order Difference of daily return')
plt.xlabel('Date')
plt.ylabel('2nd Differnece of return')


# ## ACF

# In[197]:


acf_vals = acf(d1return, nlags=20)
fig, ax = plt.subplots(figsize=(10, 2))
plot_acf(d1return, ax=ax, lags=20, alpha=0.05)
ax.axhline(y=-1.96/np.sqrt(len(d1return)), linestyle='--', color='gray')
ax.axhline(y=1.96/np.sqrt(len(d1return)), linestyle='--', color='gray')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')
plt.title("The ACF for 1st difference")
plt.show()


# ## pacf
# 

# In[199]:


pacf_vals = pacf(d1return, nlags=20)
fig, ax = plt.subplots(figsize=(10, 2))
plot_pacf(d1return, ax=ax, lags=20, alpha=0.05)
ax.axhline(y=-1.96/np.sqrt(len(d1return)), linestyle='--', color='gray')
ax.axhline(y=1.96/np.sqrt(len(d1return)), linestyle='--', color='gray')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')
plt.title("The PACF for 1st differnece")
plt.show()


# In[147]:


acf_vals = acf(d2return, nlags=20)
fig, ax = plt.subplots(figsize=(10, 2))
plot_acf(d1return, ax=ax, lags=20, alpha=0.05)
ax.axhline(y=-1.96/np.sqrt(len(d2return)), linestyle='--', color='gray')
ax.axhline(y=1.96/np.sqrt(len(d2return)), linestyle='--', color='gray')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')
plt.title("The ACF for 2nd difference")
pacf_vals = pacf(d2return, nlags=20)
fig, ax = plt.subplots(figsize=(10, 2))
plot_pacf(d1return, ax=ax, lags=20, alpha=0.05)
ax.axhline(y=-1.96/np.sqrt(len(d2return)), linestyle='--', color='gray')
ax.axhline(y=1.96/np.sqrt(len(d2return)), linestyle='--', color='gray')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')
plt.title("The PACF for 2nd differnece")
plt.show()
plt.show()


# **AR term:0,1,2; MA term: 1**

# ## ARIMA(1,1,1)

# In[210]:


model = sm.tsa.ARIMA(d1return, order=(1,1,1))
result = model.fit()
aic = results.aic
print(result.summary())


# ## SARIMA

# In[110]:


ts = pd.Series(d1return)
seasonal_acf = sm.tsa.stattools.acf(ts.diff(12)[12:])
seasonal_pacf = sm.tsa.stattools.pacf(ts.diff(12)[12:])
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
sm.graphics.tsa.plot_acf(seasonal_acf, lags=14, ax=ax[0])
sm.graphics.tsa.plot_pacf(seasonal_pacf, lags=14, ax=ax[1])
for a in ax:
    a.axhline(0, linestyle='--', color='grey', alpha=0.5)
    a.axhline(0.05, linestyle='--', color='red', alpha=0.5)
    a.axhline(-0.05, linestyle='--', color='red', alpha=0.5)
plt.show()


# ## The intervention study

# In[12]:


x=data_Tesla[["Drift","dummy"]]
y=data_Tesla[["Daily return"]]
x= sm.add_constant(x)
model = sm.OLS(y, x).fit()

print(model.summary())


# ## Arch_Lm test

# In[11]:


model = arch_model(d1return, vol='GARCH', p=1, q=1)
results = model.fit()

lm_test = results.arch_lm_test()
print(lm_test)


# ## GARCH(1,1)

# In[25]:


model = arch_model(d1return, mean='Zero', vol='GARCH', p=1, q=1)
results = model.fit()
forecast = res.forecast(horizon=30)
print(results.summary)
sm.stats.acorr_ljungbox(results.resid, lags=[10], return_df=True)
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(results.conditional_volatility)

plt.title('GARCH Conditional Volatility')
plt.subplot(212)
plt.plot(results.resid)
plt.title('GARCH Residuals')
plt.show()


# ## GARCH(1,2)

# In[16]:


model = arch_model(d1return, mean='Zero', vol='GARCH', p=1, q=2)
results = model.fit()
print(results.summary)


# ## GARCH(2,2)

# In[17]:


model = arch_model(d1return, mean='Zero', vol='GARCH', p=2, q=2)
results = model.fit()
print(results.summary)


# ## GARCH(2,1)

# In[18]:


model = arch_model(d1return, mean='Zero', vol='GARCH', p=2, q=1)
results = model.fit()
print(results.summary)


# ## Residual analysis for GARCH(1,1)

# In[13]:


fig = sm.qqplot(results.resid, line='s')
plt.title('Residual QQ plot')
plt.xlabel('Theoretical quantiles')
plt.ylabel('Sample quantiles')
plt.show()


# In[142]:


pacf_vals = pacf(results.resid, nlags=20)
fig, ax = plt.subplots(figsize=(10, 2))
plot_pacf(results.resid, ax=ax, lags=20, alpha=0.05)
ax.axhline(y=-1.96/np.sqrt(len(results.resid)), linestyle='--', color='gray')
ax.axhline(y=1.96/np.sqrt(len(results.resid)), linestyle='--', color='gray')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')
plt.show()

