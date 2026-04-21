import pandas as pd

# Load merged data
df = pd.read_csv('merged_stock_rainfall.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Sort by Date to avoid future issues
df = df.sort_values('Date')

# 1️⃣ Lag Features (Previous month's value)
df['Rainfall_Lag1'] = df['Rainfall'].shift(1)
df['Close_Lag1'] = df['Close'].shift(1)

# 2️⃣ Percentage Change in Stock Price
df['Price_Change_Pct'] = df['Close'].pct_change() * 100  # in percentage

# 3️⃣ Rolling Averages (3-month)
df['Rainfall_MA_3'] = df['Rainfall'].rolling(window=3).mean()
df['Close_MA_3'] = df['Close'].rolling(window=3).mean()

# 4️⃣ Rainfall Anomaly (how much current differs from average)
rainfall_mean = df['Rainfall'].mean()
df['Rainfall_Anomaly'] = df['Rainfall'] - rainfall_mean

# 🧹 Optional: Drop rows with NaNs created by shifting/rolling
df = df.dropna().reset_index(drop=True)

# Lag features
df['Close_t-1'] = df['Close'].shift(1)
df['Close_t-2'] = df['Close'].shift(2)
df['Rainfall_t-1'] = df['Rainfall'].shift(1)

# Drop rows with NaN (due to shift)
df = df.dropna().reset_index(drop=True)

# Save updated dataset
df.to_csv('engineered_features.csv', index=False)
print("✅ Lag features added and file saved as 'engineered_features.csv'")
print(df.head())

