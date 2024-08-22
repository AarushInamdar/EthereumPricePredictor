import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '${:,.2f}'.format

# Fetch Ethereum data from Yahoo Finance
today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'
eth_df = yf.download('ETH-USD', start_date, today)
eth_df.reset_index(inplace=True)

# Inspect and clean the data
eth_df.info()

# Check for missing values
print("Missing values in dataset:", eth_df.isnull().sum())

# Fill missing values using forward fill method
eth_df.fillna(method='ffill', inplace=True)

# Apply log transformation to stabilize variance
eth_df['log_Open'] = np.log(eth_df['Open'])

# Apply Box-Cox transformation for normalization
eth_df['boxcox_Open'], lam = boxcox(eth_df['Open'])

# Augmented Dickey-Fuller test for stationarity
adf_result = adfuller(eth_df['Open'])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')

# Autocorrelation and Partial Autocorrelation plots
plot_acf(eth_df['Open'])
plot_pacf(eth_df['Open'])
plt.show()

# Data preparation for Prophet
df = eth_df[["Date", "Open"]]
new_names = {"Date": "ds", "Open": "y"}
df.rename(columns=new_names, inplace=True)

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Plot the original time series data
x = df["ds"]
y = df["y"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y))
fig.update_layout(title_text="Time Series Plot of Ethereum Open Price", xaxis=dict(rangeselector=dict(buttons=list([dict(count=1, label="1m", step="month", stepmode="backward"), dict(count=6, label="6m", step="month", stepmode="backward"), dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label="1y", step="year", stepmode="backward"), dict(step="all"),])), rangeslider=dict(visible=True), type="date",))
fig.show()

# Create and fit the Prophet model with additional parameters
m = Prophet(
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=10.0
)

# Add holidays and other special events
holidays = pd.DataFrame({
    'holiday': 'crypto_boom',
    'ds': pd.to_datetime(['2017-12-17', '2021-01-07', '2021-11-09']),
    'lower_window': 0,
    'upper_window': 1,
})
m.add_country_holidays(country_name='US')
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.add_regressor('boxcox_Open')
m.fit(train_df)

# Forecast the future and include confidence intervals
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Evaluate model performance on test set
test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
mse = mean_squared_error(test_df['y'], test_forecast['yhat'])
mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])
rmse = np.sqrt(mse)
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Plot forecast results
plot_plotly(m, forecast)
plot_components_plotly(m, forecast)

# Technical Indicators: Moving Averages, RSI, MACD
eth_df['MA50'] = eth_df['Close'].rolling(window=50).mean()
eth_df['MA200'] = eth_df['Close'].rolling(window=200).mean()

# Calculate RSI
delta = eth_df['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
eth_df['RSI'] = 100 - (100 / (1 + rs))

# Calculate MACD
eth_df['EMA12'] = eth_df['Close'].ewm(span=12, adjust=False).mean()
eth_df['EMA26'] = eth_df['Close'].ewm(span=26, adjust=False).mean()
eth_df['MACD'] = eth_df['EMA12'] - eth_df['EMA26']
eth_df['Signal'] = eth_df['MACD'].ewm(span=9, adjust=False).mean()

# Plot Technical Indicators
plt.figure(figsize=(14,7))
plt.subplot(3, 1, 1)
plt.plot(eth_df['ds'], eth_df['Close'], label='Close')
plt.plot(eth_df['ds'], eth_df['MA50'], label='50-Day MA')
plt.plot(eth_df['ds'], eth_df['MA200'], label='200-Day MA')
plt.title('Ethereum Close Price with Moving Averages')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(eth_df['ds'], eth_df['RSI'], label='RSI')
plt.axhline(70, color='r', linestyle='--')
plt.axhline(30, color='g', linestyle='--')
plt.title('Relative Strength Index (RSI)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(eth_df['ds'], eth_df['MACD'], label='MACD')
plt.plot(eth_df['ds'], eth_df['Signal'], label='Signal Line')
plt.title('Moving Average Convergence Divergence (MACD)')
plt.legend()

plt.tight_layout()
plt.show()

# Model Fine-Tuning: Grid Search for Hyperparameters
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0],
    'seasonality_prior_scale': [1.0, 5.0, 10.0, 20.0],
}

best_params = None
best_rmse = float('inf')

for cps in param_grid['changepoint_prior_scale']:
    for sps in param_grid['seasonality_prior_scale']:
        m = Prophet(
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        m.fit(train_df)
        future = m.make_future_dataframe(periods=len(test_df))
        forecast = m.predict(future)
        test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
        rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
        print(f'Params: cps={cps}, sps={sps} -> RMSE: {rmse}')
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (cps, sps)

print(f'Best parameters: cps={best_params[0]}, sps={best_params[1]} with RMSE: {best_rmse}')

# Retrain the model with the best parameters
m = Prophet(
    changepoint_prior_scale=best_params[0],
    seasonality_prior_scale=best_params[1],
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
m.fit(train_df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
plot_plotly(m, forecast)

# Model Export: Save the model for future use
import joblib
joblib.dump(m, 'prophet_eth_model.pkl')

# Load the model later
loaded_model = joblib.load('prophet_eth_model.pkl')
loaded_forecast = loaded_model.predict(future)
plot_plotly(loaded_model, loaded_forecast)

# Add Custom Holiday Effects
eth_df['event'] = eth_df['ds'].apply(lambda x: 1 if x in ['2017-12-17', '2021-01-07', '2021-11-09'] else 0)
eth_df['y'] = eth_df['Open'] + eth_df['event'] * 100  # Hypothetical impact of event
df_event = eth_df[['ds', 'y', 'event']]
m = Prophet()
m.add_regressor('event')
m.fit(df_event)
future_event = m.make_future_dataframe(periods=365)
future_event['event'] = 0  # No events in the future
forecast_event = m.predict(future_event)
plot_plotly(m, forecast_event)

# Analyze Residuals for Model Diagnostics
eth_df['residual'] = eth_df['y'] - forecast['yhat'].values[:len(eth_df)]
plt.figure(figsize=(10,5))
plt.plot(eth_df['ds'], eth_df['residual'])
plt.title('Residuals of the Model')
plt.show()

# Histogram and QQ plot for residuals
sm.qqplot(eth_df['residual'], line ='45')
plt.show()
plt.hist(eth_df['residual'], bins=50)
plt.title('Histogram of Residuals')
plt.show()
