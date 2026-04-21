import yfinance as yf
import pandas as pd

# Download stock data
stock = yf.download('RELIANCE.NS', start='2015-01-01', end='2023-12-31', interval='1mo', auto_adjust=False)

# Reset index to make 'Date' a column
stock = stock.reset_index()

# Flatten any multi-level columns
stock.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in stock.columns]

# Get actual column names
date_col = [col for col in stock.columns if 'Date' in col][0]
close_col = [col for col in stock.columns if 'Close' in col][0]

# Rename columns
stock = stock.rename(columns={date_col: 'Date', close_col: 'Close'})

# Convert 'Date' to datetime
stock['Date'] = pd.to_datetime(stock['Date'])

# Load rainfall data
rainfall = pd.read_csv('rainfall_data.csv')
rainfall['Date'] = pd.to_datetime(rainfall['Date'])

# Merge on 'Date'
merged_df = pd.merge(stock[['Date', 'Close']], rainfall, on='Date', how='inner')

# Sort by date
merged_df = merged_df.sort_values('Date')

# Add lag features
num_lags = 3

# Lag for stock Close
for lag in range(1, num_lags + 1):
    merged_df[f'Close_lag_{lag}'] = merged_df['Close'].shift(lag)

# Lag for Rainfall
for lag in range(1, num_lags + 1):
    merged_df[f'Rainfall_lag_{lag}'] = merged_df['Rainfall'].shift(lag)

# Drop rows with NaN values (from shifting)
merged_df = merged_df.dropna().reset_index(drop=True)

# Save to new CSV
merged_df.to_csv('merged_with_lags.csv', index=False)

print("✅ Final dataset with lag features saved as 'merged_with_lags.csv'")
print(merged_df.head())
