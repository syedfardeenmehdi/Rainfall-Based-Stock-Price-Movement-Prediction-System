import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load feature-engineered dataset
df = pd.read_csv('engineered_features.csv')

# Drop rows with any missing values (important for lag/MAs)
df.dropna(inplace=True)

# Define features and target
features = ['Rainfall', 'Close_t-1', 'Rainfall_t-1', 'Rainfall_MA_3', 'Close_MA_3', 'Rainfall_Anomaly']
X = df[features]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Save predictions
results = df[['Date']].iloc[-len(y_test):].copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred
results.to_csv('rf_model_predictions.csv', index=False)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📉 Mean Squared Error (Random Forest): {mse}")
print(f"📈 R² Score (Random Forest): {r2}")
print("✅ Predictions saved to rf_model_predictions.csv")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Stock Prices (Random Forest)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
