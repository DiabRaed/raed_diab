#%%
import yfinance as yf
import matplotlib.pyplot as plt
# Download historical data for a stock (e.g., Apple)
data = yf.download("AAPL", start="2010-01-01", end="2023-01-01")
print(data.head())

#let's consider moving average 
data['SMA_50'] = data['Close'].rolling(window=30).mean()
plt.plot(data['Adj Close'],c='b')
plt.plot(data['SMA_50'],'--',c='orange')
# %%
#check if it's stationary using ADF
# The Augmented Dickey-Fuller (ADF) test is a statistical test used to determine whether a time series is stationary. Stationarity is a crucial concept in time-series analysis because many models (like ARIMA) assume that the input data is stationary.

# What is Stationarity?
# A time series is stationary if its statistical properties (mean, variance, autocorrelation, etc.) do not change over time. In other words:

# The mean and variance are constant over time.
# The covariance between time points depends only on the lag, not the time itself.
# Non-stationary data often exhibit trends, seasonality, or other time-dependent patterns.

# Why Test for Stationarity?
# If your time series is non-stationary, you might need to transform it (e.g., via differencing, detrending, or seasonal decomposition) before applying certain models like ARIMA.

from statsmodels.tsa.stattools import adfuller
import pandas as pd
# Perform the ADF test on the closing prices
adf_result = adfuller(data['Close'].dropna())

# Visualize the closing prices
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Apple Stock Prices')
plt.title('Apple Stock Prices (2010-2023)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# ADF test results

# Print each component of the ADF result with explanations
print(f"ADF Statistic: {adf_result[0]} (Test statistic, compare with critical values)")
print(f"p-value: {adf_result[1]} (Probability of the null hypothesis being true)")
print(f"Number of Lags Used: {adf_result[2]} (Lagged differences used in the test)")
print(f"Number of Observations Used: {adf_result[3]} (Data points considered after lag adjustments)")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"  {key}: {value} (Critical value for {key} confidence level)")
print(f"Maximized Information Criterion: {adf_result[5]} (Useful for model comparison)")
#%%
#because p-value is bigger than 0.05, the series is likely
#non-stationary. We have to transform that to make it stationary 

data['Differenced'] = data['Close'].diff().dropna()
from statsmodels.tsa.stattools import adfuller

diff_adf_result = adfuller(data['Differenced'].dropna())
print(f"ADF Statistic: {diff_adf_result[0]}")
print(f"p-value: {diff_adf_result[1]}")
print("Critical Values:", diff_adf_result[4])


#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF and PACF
plot_acf(data['Differenced'].dropna(), lags=20)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plot_pacf(data['Differenced'].dropna(), lags=20)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()


#%%
from statsmodels.tsa.arima.model import ARIMA

# Autocorrelation (ACF) Plot:
# The x-axis represents the lags (i.e., the number of periods ago the correlation is being measured).
# The y-axis shows the correlation values.
# The lines (usually dashed) represent the confidence interval. If a correlation value is outside this band, it’s considered significant.
# Cutoff: When the correlation drops significantly (outside the confidence interval), that’s where the MA (q) parameter may end.
# Partial Autocorrelation (PACF) Plot:
# Same as ACF, but the correlation values are adjusted to account for the previous lags.
# The point where the PACF cuts off (drops to zero) typically represents the p (AR order).
# Summary of How to Choose p and q:
# For p (AR): Look at the PACF. The lag at which the PACF cuts off is a good candidate for p.
# For q (MA): Look at the ACF. The lag at which the ACF cuts off is a good candidate for q.


# Fit the ARIMA model
model = ARIMA(data['Close'], order=(0, 1, 0))
arima_result = model.fit()

# Print model summary
print(arima_result.summary())

# Forecast future values
forecast = arima_result.forecast(steps=30)  # Forecast 30 days
print(forecast)

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Actual')
# plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()

from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(data['Close'][-30:], forecast[:30])
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")



#%%
import yfinance as yf

def evaluate_stock(stock_symbol):
    # Fetch stock data using the yfinance library
    stock = yf.Ticker(stock_symbol)

    # Get stock information
    stock_info = stock.info

    # Print available keys to help understand the structure of the data
    print("Available keys in the stock data:")
    print(stock_info.keys(), end="\n\n")
    # print("The stock name is ", stock_info.longName)
    # Extract financial data with error handling
    try:
        stock_price = stock_info['currentPrice']
    except KeyError:
        stock_price = None

    try:
        eps = stock_info.get('trailingEps', None)  # Use 'trailingPE' if 'epsTrailingTwelveMonths' is not available
    except KeyError:
        eps = None

    try:
        dividends = stock_info.get('dividendYield', None)  # Using 'dividendYield' instead of 'dividendRate'
    except KeyError:
        dividends = None

    try:
        book_value = stock_info.get('bookValue', None)  # Book Value per Share
    except KeyError:
        book_value = None

    try:
        total_debt = stock_info.get('totalDebt', None)  # Total Debt
    except KeyError:
        total_debt = None

    try:
        total_equity = stock_info.get('totalEquity', None)  # Total Equity
    except KeyError:
        total_equity = None

    # Check if critical data is missing and handle appropriately
    if eps is None or stock_price is None:
        print("Critical data is missing, can't calculate ratios.")
        return None

    # Calculate Ratios
    print(f"\n### Stock Evaluation for {stock_symbol} ###\n")
    
    # Calculate P/E Ratio
    if eps:
        pe_ratio = stock_price / eps
        print(f"P/E Ratio = Stock Price / EPS = {stock_price} / {eps} = {pe_ratio}")
    else:
        pe_ratio = None
        print("P/E Ratio is unavailable.")
    
    # Calculate Dividend Yield
    if dividends:
        dividend_yield = dividends * 100
        print(f"Dividend Yield = {dividends} * 100 = {dividend_yield}%")
    else:
        dividend_yield = None
        print("Dividend Yield is unavailable.")
    
    # Calculate P/B Ratio
    if book_value:
        pb_ratio = stock_price / book_value
        print(f"P/B Ratio = Stock Price / Book Value = {stock_price} / {book_value} = {pb_ratio}")
    else:
        pb_ratio = None
        print("P/B Ratio is unavailable.")
    
    # Calculate Debt-to-Equity Ratio
    if total_debt and total_equity:
        debt_to_equity = total_debt / total_equity
        print(f"Debt-to-Equity Ratio = Total Debt / Total Equity = {total_debt} / {total_equity} = {debt_to_equity}")
    else:
        debt_to_equity = None
        print("Debt-to-Equity Ratio is unavailable.")
    # Define Evaluation Logic as a Continuous Score
    evaluation_score = 0

    # Scoring Criteria
    if pe_ratio:
        if pe_ratio < 15:
            evaluation_score += 25  # Favorable P/E
        elif pe_ratio > 25:
            evaluation_score -= 25  # Unfavorable P/E

    if dividend_yield:
        if dividend_yield > 4:
            evaluation_score += 25  # Favorable Dividend Yield
        elif dividend_yield < 2:
            evaluation_score -= 25  # Unfavorable Dividend Yield

    if pb_ratio:
        if pb_ratio < 1:
            evaluation_score += 25  # Favorable P/B
        elif pb_ratio > 3:
            evaluation_score -= 25  # Unfavorable P/B

    if debt_to_equity:
        if debt_to_equity < 1:
            evaluation_score += 25  # Favorable Debt-to-Equity
        elif debt_to_equity > 2:
            evaluation_score -= 25  # Unfavorable Debt-to-Equity

    # Map the Score to a Continuous Bar
    evaluation_score = max(-100, min(evaluation_score, 100))  # Clamp the score between -100 and 100

    # Display the Continuous Bar
    bar_length = 10  # Total length of the bar
    filled_length = int((evaluation_score + 100) / 200 * bar_length)  # Map score to bar length
    bar = "[" + "=" * filled_length + " " * (bar_length - filled_length) + "]"

    # Print Results
    print(f"\n### Stock Evaluation for {stock_symbol} ###")
    print(f"Evaluation Bar: {bar} ({evaluation_score})")
    if evaluation_score > 50:
        print("Stock is Strongly Undervalued.")
    elif evaluation_score > 0:
        print("Stock is Slightly Undervalued.")
    elif evaluation_score < -50:
        print("Stock is Strongly Overvalued.")
    elif evaluation_score < 0:
        print("Stock is Slightly Overvalued.")
    else:
        print("Stock is Neutral.")


    print(f"\nFinal Stock Evaluation: {evaluation_score}")

    # Display final results
    print("\n### Final Evaluation Results ###")
    print(f"Stock Price: ${stock_price}")
    print(f"P/E Ratio: {pe_ratio}")
    print(f"Dividend Yield: {dividend_yield}%")
    print(f"P/B Ratio: {pb_ratio}")
    print(f"Debt-to-Equity Ratio: {debt_to_equity}")
    print(f"Stock Evaluation: {evaluation_score}")

evaluate_stock("VOO")

#%%
import yfinance as yf

def evaluate_stock_and_dcf(stock_symbol, growth_rate, discount_rate, years=5):
    # Fetch stock data using yfinance
    stock = yf.Ticker(stock_symbol)
    stock_info = stock.info

    # Get the current market price
    current_market_price = stock_info.get('currentPrice')
    if current_market_price is None:
        print("Current Market Price is not available.")
        return None

    # Get the current free cash flow
    fcf = stock_info.get('freeCashflow')
    if fcf is None:
        print("Free Cash Flow is not available.")
        return None

    # Project future free cash flows
    projected_fcfs = []
    for i in range(years):
        projected_fcf = fcf * ((1 + growth_rate) ** (i + 1))
        projected_fcfs.append(projected_fcf)

    # Calculate the present value of future cash flows
    present_value = sum(projected_fcf / ((1 + discount_rate) ** (i + 1)) for i, projected_fcf in enumerate(projected_fcfs))

    # Estimate terminal value using Gordon Growth Model
    terminal_value = projected_fcfs[-1] * (1 + growth_rate) / (discount_rate - growth_rate)
    present_value_terminal = terminal_value / ((1 + discount_rate) ** years)

    # Total intrinsic value
    intrinsic_value = present_value + present_value_terminal

    # Get the number of shares outstanding
    shares_outstanding = stock_info.get('sharesOutstanding', 1)  # Avoid division by zero

    # Calculate intrinsic value per share
    intrinsic_value_per_share = intrinsic_value / shares_outstanding

    # Print results
    print(f"\n### DCF Evaluation for {stock_symbol} ###")
    print(f"Current Market Price: ${current_market_price:.2f}")
    print(f"Current Free Cash Flow: ${fcf:.2f}")
    print(f"Projected Free Cash Flows: {projected_fcfs}")
    print(f"Intrinsic Value: ${intrinsic_value:.2f}")
    print(f"Intrinsic Value Per Share: ${intrinsic_value_per_share:.2f}")

    # Evaluate on a scale from 0 to 10
    evaluation_score = 0

    if intrinsic_value_per_share < current_market_price:
        # Overvalued
        evaluation_score = max(0, 10 * (1 - (intrinsic_value_per_share / current_market_price)))
    elif intrinsic_value_per_share > current_market_price:
        # Undervalued
        evaluation_score = min(10, 10 * (intrinsic_value_per_share / current_market_price))
    else:
        # Fairly valued
        evaluation_score = 5

    # Print evaluation score
    if evaluation_score == 5:
        print("The stock is fairly valued.")
    elif evaluation_score > 5:
        print(f"The stock is undervalued with a score of {evaluation_score:.2f}/10.")
    else:
        print(f"The stock is overvalued with a score of {evaluation_score:.2f}/10.")

    return intrinsic_value_per_share

# Example usage
evaluate_stock_and_dcf("AAPL", growth_rate=0.05, discount_rate=0.08)


#%% 
#piece of code to calculate APY 

#assume starting with $1k 
start_amount=9000

current_APY=0.035


net_amount=start_amount+start_amount*current_APY
print(net_amount)
#%%
#But, if I have contributions, then I have to take that 
# into account  

#Let's assume that I add $200 every 14 days.
# for a month
daily_APY=current_APY/365 
contribution=200
net_amount=((start_amount+daily_APY*start_amount*14)
+contribution+(start_amount+contribution)*daily_APY*14)

print(net_amount)
daily_APY = (1 + current_APY) ** (1 / 365) - 1 #this is a more accurate one

#Now, let's do this for 12 months 
current_amount=1000
added_money=200
monthly_amount=[]
num_years=10
period=num_years*365
months_passed=[]
total_contribution=0
for i in range(period):
    # print(i)
    current_amount += daily_APY*current_amount
    if (i+1) % 14 ==0:
        current_amount += added_money
        total_contribution+=added_money

    if (i+1) % 30 ==0: 
        # print("This is the end of the month savings:", current_amount)
        monthly_amount.append(current_amount)
        months_passed.append(i)
    
print(f"Over {np.ceil(np.array(months_passed)/365)[-1]} years, I have:")
print(f"This is the final amount is: ${(current_amount/1e3):.2f}k")
print(f"Total contribution is: ${(total_contribution/1e3):.2f}k")

plt.plot(np.array(months_passed)/365,monthly_amount,label='Savings ')
plt.xlabel("# of Years  ")
plt.ylabel("Amount in USD")
plt.title("Savings calculation")
plt.legend()
# %%
#to make this interesting, assuming a changing APY 
periods = [80, 40, 90, 100, 55]  # Days in each period (should sum to 365)
apy_values = [0.043, 0.0425, 0.042, 0.041, 0.04]  # Corresponding APYs

# Generate the APY list
apy_list = []
for days, apy in zip(periods, apy_values):
    apy_list.extend([apy] * days)

# daily_APY=np.array(apy_list)/365
daily_APY = (1 + np.array(apy_list)) ** (1 / 365) - 1 #this is a more accurate one

current_amount=1000
monthly_amount=[]
num_years=1
period=num_years*365
months_passed=[]
total_contribution=0
for i in range(period):
    # print(i,idx)
    current_amount += daily_APY[i]*current_amount
    if (i+1) % 14 ==0:
        current_amount += 200
        total_contribution+=200
    if (i+1) % 30 ==0: 
        # print("This is the end of the month savings:", current_amount)
        monthly_amount.append(current_amount)
        months_passed.append(i)
print(f"This is the final amount in USD: {current_amount:.2f}")
print(f"Total contribution is: {total_contribution:.2f}")
plt.plot(np.array(months_passed)/365,monthly_amount,label='Savings ')
plt.xlabel("# of Years  ")
plt.ylabel("Amount in USD")
plt.title("Savings calculation")
plt.legend()

# %%


from PIL import Image

# # Load the original image
# img = Image.open("/Users/username/Downloads/new_image.jpg")

# # Resize to 1.96x2.755 inches at 300 DPI (588x827 pixels)
# target_size = (588, 827)
# img_resized = img.resize(target_size, Image.LANCZOS)

# # Create a blank 4x6-inch image (1200x1800 pixels) with a light gray background
# canvas_size = (1200, 1800)
# canvas_color = (224, 224, 224)  # Light gray (RGB: #E0E0E0)
# canvas = Image.new("RGB", canvas_size, canvas_color)

# # Calculate position to center the image
# x_offset = (canvas_size[0] - target_size[0]) // 2
# y_offset = (canvas_size[1] - target_size[1]) // 2

# # Paste the resized image onto the canvas
# canvas.paste(img_resized, (x_offset, y_offset))

# # Save the final image for printing
# canvas.save("/Users/username/Downloads/final_4x6_custom_bg.jpg", dpi=(300, 300))



# Load the original image
img = Image.open("/Users/username/Downloads/image.jpg")

# Convert 35mm x 45mm to pixels at 300 DPI
# 1 inch = 25.4 mm
# 35mm = 35/25.4 = 1.378 inches, at 300 DPI = 413 pixels
# 45mm = 45/25.4 = 1.772 inches, at 300 DPI = 532 pixels
target_size = (413, 532)  # 35mm x 45mm at 300 DPI

# Resize with high-quality resampling
img_resized = img.resize(target_size, Image.LANCZOS)

# Create a blank 4x6-inch canvas at 300 DPI (1200x1800 pixels) with light gray background
canvas_size = (1200, 1800)
canvas_color = (224, 224, 224)  # Light gray (RGB: #E0E0E0)
canvas = Image.new("RGB", canvas_size, canvas_color)

# Calculate position to center the image
x_offset = (canvas_size[0] - target_size[0]) // 2
y_offset = (canvas_size[1] - target_size[1]) // 2

# Paste the resized image onto the canvas
canvas.paste(img_resized, (x_offset, y_offset))

# Save the final image for printing
canvas.save("/Users/username/Downloads/final_4x6_passport_photo.jpg", quality=95)


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Fetch Apple stock data
ticker = "AAPL"
start_date = "2023-01-01"
end_date = "2025-03-01"
stock_data = yf.download(ticker, start=start_date, end=end_date)
closing_prices = stock_data["Close"].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Create sequences
seq_length = 30 #the number of days trained to predict the next day
X, y = create_sequences(scaled_data, seq_length)

# Define the test period
test_start_date = pd.to_datetime("2025-02-02")
test_end_date = pd.to_datetime("2025-02-22")

# Find the indices for the test period
test_start_index = (stock_data.index >= test_start_date).argmax() - seq_length
test_end_index = (stock_data.index >= test_end_date).argmax()

# Validate indices
if test_start_index < 0:
    test_start_index = 0
if test_end_index <= test_start_index:
    test_end_index = test_start_index + 1

print(f"Test Start Date: {test_start_date}, Index: {test_start_index}")
print(f"Test End Date: {test_end_date}, Index: {test_end_index}")

# Split into training and testing sets
X_train = X[:test_start_index]
y_train = y[:test_start_index]
X_test = X[test_start_index:test_end_index]
y_test = y[test_start_index:test_end_index]

print("X_test shape:", X_test.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=50, verbose=0)

# Make predictions for the test period
predictions_test = model.predict(X_test)
predictions_test = scaler.inverse_transform(predictions_test)
y_test = scaler.inverse_transform(y_test)

# Plotting the test period predictions
test_plot_start = test_start_index + seq_length
test_plot_end = test_end_index
plt.figure(figsize=(12, 6))

# Debugging prints:
print(f"Date range shape: {stock_data.index[test_plot_start:test_plot_end].shape}")
print(f"y_test shape: {y_test.shape}")
print(f"predictions_test shape: {predictions_test.shape}")

# Corrected slicing to match the length of predictions
actual_prices_plot = y_test[:test_end_index - test_plot_start]
predicted_prices_plot = predictions_test[:test_end_index - test_plot_start]

print(f"actual_prices_plot shape: {actual_prices_plot.shape}")
print(f"predicted_prices_plot shape: {predicted_prices_plot.shape}")

plt.plot(stock_data.index[test_plot_start:test_plot_end], actual_prices_plot, label='Actual Prices (Feb 2-20)')
plt.plot(stock_data.index[test_plot_start:test_plot_end], predicted_prices_plot, label='Predicted Prices (Feb 2-20)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Apple Stock Price Prediction vs. Actual (Feb 2-20)')
plt.legend()
plt.grid(True)
plt.show()

# Predict for March 2025
future_steps = 5 #The number of days predicted ahead. it uses the number of days above (30) to predict day 1
last_sequence = scaled_data[-seq_length:]
future_predictions = []

for i in range(future_steps):
    sequence_input = last_sequence.reshape((1, seq_length, 1))
    prediction = model.predict(sequence_input)
    future_predictions.append(scaler.inverse_transform(prediction)[0, 0])
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)

#%%
future_dates = pd.date_range(start="2025-02-24", periods=future_steps)

# Plotting March 2025 predictions
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Historical Data')
plt.plot(future_dates, future_predictions, label='Predicted Prices (March 2025)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Apple Stock Price Prediction (March 2025)')
plt.legend()
plt.grid(True)

start_xlim = pd.to_datetime("2024-10-01") # Define your start xlim
end_xlim = pd.to_datetime("2025-03-31")   # Define your end xlim
plt.xlim(start_xlim, end_xlim)

plt.show()
# %%

import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# List of stock tickers to track
stocks = ['AAPL', 'MSFT', 'KO', 'PEP', 'JNJ', 'PG', 'TGT', 'GE', 'NVDA', 'O']

# Define the time period (last 6 months as an example)
start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Fetch the data
def fetch_data(stock_list):
    stock_data = {}
    for stock in stock_list:
        data = yf.download(stock, start=start_date, end=end_date)
        stock_data[stock] = data
    return stock_data

# Get stock data
stock_data = fetch_data(stocks)

# Display the fetched data for a single stock
print(stock_data['AAPL'].tail())  # Example: Show the last 5 rows of Apple's data

# Function to calculate moving averages and volatility
def calculate_indicators(stock_data):
    stock_indicators = {}
    
    for stock, data in stock_data.items():
        # Calculate 50-day moving average
        data['50ma'] = data['Close'].rolling(window=50).mean()
        
        # Calculate 200-day moving average
        data['200ma'] = data['Close'].rolling(window=200).mean()
        
        # Calculate daily returns
        data['Daily Return'] = data['Close'].pct_change()
        
        # Calculate volatility (standard deviation of daily returns)
        data['Volatility'] = data['Daily Return'].rolling(window=50).std()
        
        # Add to dictionary
        stock_indicators[stock] = data

    return stock_indicators

# Calculate the indicators
stock_indicators = calculate_indicators(stock_data)

# Display the most recent data for Apple, including the new indicators
print(stock_indicators['AAPL'].tail())


def rank_stocks(stock_indicators):
    stock_scores = []

    for stock, data in stock_indicators.items():
        latest_data = data.iloc[-1]  # Get the last row
        
        # Ensure 'Close' price is scalar
        latest_close = latest_data['Close']
        
        # Check if 50ma is available for comparison
        latest_50ma = data['50ma'].iloc[-1] if not np.isnan(data['50ma'].iloc[-1]) else None
        
        # Skip this stock if there's no valid 50ma
        if latest_50ma is None:
            continue
        
        # Initialize the score
        score = 0
        
        # Check if the stock is above the 50-day moving average
        if latest_50ma is not None and latest_close > latest_50ma:
            score += 1
        
        # Check if the stock has low volatility
        latest_volatility = latest_data['Volatility']
        if latest_volatility < data['Volatility'].mean():
            score += 1
        
        # Check for strong price growth (last 6 months)
        growth = (latest_close / data['Close'].iloc[0] - 1) * 100
        if growth > 10:  # Example: Only rank stocks that have grown by more than 10% in 6 months
            score += 1
        
        # Add stock and score to list
        stock_scores.append((stock, score))
    
    # Sort stocks by score in descending order
    ranked_stocks = sorted(stock_scores, key=lambda x: x[1], reverse=True)
    
    return ranked_stocks

# Rank the stocks
ranked_stocks = rank_stocks(stock_indicators)

# Display the ranked stocks
print("Ranked Stocks Based on Criteria:")
for stock, score in ranked_stocks:
    print(f"{stock}: Score {score}")



import matplotlib.pyplot as plt

# Plot the last 6 months of stock data for a selected stock (e.g., AAPL)
def plot_stock(stock_data, stock):
    data = stock_data[stock]
    
    plt.figure(figsize=(10,6))
    plt.plot(data['Close'], label='Closing Price')
    plt.plot(data['50ma'], label='50-Day Moving Average')
    plt.plot(data['200ma'], label='200-Day Moving Average')
    plt.title(f"{stock} - Stock Price & Moving Averages")
    plt.legend()
    plt.show()

# Plot Apple stock
plot_stock(stock_indicators, 'AAPL')

# %%
