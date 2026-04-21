import yfinance as yf
import pandas as pd

# Download data
stock = yf.download('RELIANCE.NS', start='2015-01-01', end='2023-12-31', interval='1mo', auto_adjust=False)

# Reset index to make 'Date' a column
stock = stock.reset_index()

# Flatten any multi-level columns
stock.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in stock.columns]

# Print to inspect actual column names
print("Stock columns after flattening:", stock.columns)

# Now rename columns correctly
# This will automatically work no matter what the ticker symbol is
date_col = [col for col in stock.columns if 'Date' in col][0]
close_col = [col for col in stock.columns if 'Close' in col][0]

stock = stock.rename(columns={date_col: 'Date', close_col: 'Close'})

# Convert 'Date' to datetime
stock['Date'] = pd.to_datetime(stock['Date'])

# Load rainfall data
rainfall = pd.read_csv('rainfall_data.csv')
rainfall['Date'] = pd.to_datetime(rainfall['Date'])

# Merge on Date
merged_df = pd.merge(stock[['Date', 'Close']], rainfall, on='Date', how='inner')

# Save merged data
merged_df.to_csv('merged_stock_rainfall.csv', index=False)

print("✅ Merged data saved successfully.")
print(merged_df.head())
