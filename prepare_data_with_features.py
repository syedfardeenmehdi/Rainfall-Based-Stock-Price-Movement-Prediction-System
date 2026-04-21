import pandas as pd

# Load lagged dataset
data = pd.read_csv("merged_with_lags.csv")

# --- Feature 1: Percentage Change from Last Month ---
data['Close_pct_change'] = (data['Close'] - data['Close_lag_1']) / data['Close_lag_1']

# --- Feature 2: 3-Month Rolling Average of Close Price ---
data['Close_roll_mean_3'] = data['Close'].rolling(window=3).mean()

# --- Feature 3: Interaction: Close_lag_1 × Rainfall ---
data['Rainfall_Close_interaction'] = data['Close_lag_1'] * data['Rainfall']

# Drop any new NaNs (due to rolling mean)
data = data.dropna().reset_index(drop=True)

# Save enhanced dataset
data.to_csv("enhanced_data_for_rf.csv", index=False)

print("✅ New features added and saved to 'enhanced_data_for_rf.csv'")
print(data.head())
