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

# Import additional libraries for advanced features
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from prophet.diagnostics import cross_validation, performance_metrics
import seaborn as sns

# Fetch real-time cryptocurrency news and perform sentiment analysis
def fetch_crypto_news():
    url = 'https://newsapi.org/v2/everything?q=cryptocurrency&apiKey=YOUR_NEWSAPI_KEY'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data['articles']
    return articles

def analyze_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for article in articles:
        sentiment = analyzer.polarity_scores(article['title'])
        sentiments.append(sentiment['compound'])
    avg_sentiment = np.mean(sentiments)
    return avg_sentiment

# Incorporate sentiment analysis as an additional feature in the dataset
articles = fetch_crypto_news()
sentiment_score = analyze_sentiment(articles)
eth_df['sentiment'] = sentiment_score
df['sentiment'] = sentiment_score

# Scale features before training the model
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['y', 'sentiment']])
df['scaled_y'] = scaled_features[:, 0]
df['scaled_sentiment'] = scaled_features[:, 1]

# Feature Engineering: Add Rolling Statistics
eth_df['rolling_mean_30'] = eth_df['Open'].rolling(window=30).mean()
eth_df['rolling_std_30'] = eth_df['Open'].rolling(window=30).std()

# Principal Component Analysis (PCA) for dimensionality reduction
pca = PCA(n_components=1)
eth_df['pca_component'] = pca.fit_transform(scaled_features)

# Retrain Prophet model with additional regressors and PCA component
m = Prophet(
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_prior_scale=best_params[1]
)
m.add_regressor('sentiment')
m.add_regressor('pca_component')
m.add_regressor('rolling_mean_30')
m.add_regressor('rolling_std_30')
m.fit(train_df)

# Update forecast with new features
future = m.make_future_dataframe(periods=365)
future['sentiment'] = sentiment_score
future['pca_component'] = pca.transform(scaler.transform(future[['y', 'sentiment']]))[:, 0]
future['rolling_mean_30'] = eth_df['rolling_mean_30']
future['rolling_std_30'] = eth_df['rolling_std_30']
forecast = m.predict(future)
plot_plotly(m, forecast)

# Cross-validation for model performance evaluation
cv_results = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
performance_df = performance_metrics(cv_results)
print(performance_df.head())

# Visualizing Cross-Validation Results
fig = plot_cross_validation_metric(cv_results, metric='rmse')
plt.show()

# Additional Evaluation Metrics
mape = np.mean(np.abs((test_df['y'] - test_forecast['yhat']) / test_df['y'])) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

# Enhanced Visualizations: Correlation Heatmap
correlation_matrix = eth_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Ethereum Features')
plt.show()

# Time-Series Decomposition for Trend, Seasonality, and Residuals
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(eth_df['Open'], model='multiplicative', period=365)
decompose_result.plot()
plt.show()

# Implement a custom ensemble method by combining predictions from Prophet and a simple ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Train ARIMA model on the same data
arima_model = ARIMA(train_df['y'], order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Generate forecasts with ARIMA
arima_forecast = arima_model_fit.forecast(steps=len(test_df))

# Combine Prophet and ARIMA predictions
ensemble_forecast = (forecast['yhat'][:len(test_df)] + arima_forecast) / 2

# Evaluate the ensemble model
ensemble_rmse = np.sqrt(mean_squared_error(test_df['y'], ensemble_forecast))
print(f'Ensemble Model RMSE: {ensemble_rmse}')

# Visualize ensemble forecast versus actual
plt.figure(figsize=(10, 5))
plt.plot(test_df['ds'], test_df['y'], label='Actual')
plt.plot(test_df['ds'], ensemble_forecast, label='Ensemble Forecast')
plt.title('Ensemble Model: Prophet + ARIMA')
plt.legend()
plt.show()

# Model Retraining on Full Data for Deployment
m_final = Prophet(
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_prior_scale=best_params[1]
)
m_final.add_regressor('sentiment')
m_final.add_regressor('pca_component')
m_final.add_regressor('rolling_mean_30')
m_final.add_regressor('rolling_std_30')
m_final.fit(df)  # Fit on full dataset
final_future = m_final.make_future_dataframe(periods=365)
final_future['sentiment'] = sentiment_score
final_future['pca_component'] = pca.transform(scaler.transform(final_future[['y', 'sentiment']]))[:, 0]
final_future['rolling_mean_30'] = eth_df['rolling_mean_30']
final_future['rolling_std_30'] = eth_df['rolling_std_30']
final_forecast = m_final.predict(final_future)

# Final Visualizations and Export
plot_plotly(m_final, final_forecast)
plot_components_plotly(m_final, final_forecast)

# Export final model and forecast for deployment
joblib.dump(m_final, 'final_prophet_eth_model.pkl')
final_forecast.to_csv('final_eth_forecast.csv', index=False)
