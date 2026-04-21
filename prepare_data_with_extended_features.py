import pandas as pd

# Load the merged dataset with Date, Close, and Rainfall
data = pd.read_csv("merged_stock_rainfall.csv")

# Ensure Date is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort by Date
data = data.sort_values('Date')

# Add lag features for Rainfall and Close price
for lag in range(1, 7):  # Lag up to 6 months
    data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
    data[f'Rainfall_lag_{lag}'] = data['Rainfall'].shift(lag)

# Add rolling means (optional)
data['Close_roll_mean_3'] = data['Close'].rolling(window=3).mean()
data['Rainfall_roll_mean_3'] = data['Rainfall'].rolling(window=3).mean()

# Add month and season
data['Month'] = data['Date'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Autumn'

data['Season'] = data['Month'].apply(get_season)

# Convert Season to numerical
data = pd.get_dummies(data, columns=['Season'], drop_first=True)

# Drop rows with NaNs due to shifting/rolling
data = data.dropna().reset_index(drop=True)

# Save the enhanced dataset
data.to_csv("extended_features_dataset.csv", index=False)
print("✅ Extended dataset with lag, month, and season features saved to 'extended_features_dataset.csv'")
