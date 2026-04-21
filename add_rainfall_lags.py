import pandas as pd

# Load the dataset
df = pd.read_csv('extended_features_dataset.csv')

# Ensure 'Rainfall' column exists
if 'Rainfall' not in df.columns:
    raise ValueError("❌ 'Rainfall' column not found in the dataset.")

# Generate Rainfall lag features (1 to 5)
for lag in range(1, 6):
    df[f'Rainfall_lag_{lag}'] = df['Rainfall'].shift(lag)

# Drop rows with NaN due to lagging
df.dropna(inplace=True)

# Save updated dataset
df.to_csv('extended_features_dataset.csv', index=False)
print("✅ Rainfall lag features added and saved to 'extended_features_dataset.csv'")
