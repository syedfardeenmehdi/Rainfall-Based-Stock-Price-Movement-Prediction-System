import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('extended_features_dataset.csv')

# Define target and selected important features
target = 'Close'
important_features = [
    'Close_roll_mean_3',
    'Close_lag_1',
    'Close_lag_6',
    'Close_lag_3',
    'Close_lag_2',
    'Close_lag_4',
    'Close_lag_5'
]

X = data[important_features]
y = data[target]

# Train-test split (80/20, no shuffle to preserve time series order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📉 Mean Squared Error (Random Forest): {mse}")
print(f"📈 R² Score (Random Forest): {r2}")

# Save predictions
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred
results.to_csv('rf_model_predictions_precise.csv', index=False)
print("✅ Predictions saved to 'rf_model_predictions_precise.csv'")
import matplotlib.pyplot as plt

# Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(results['Actual'].values, label='Actual', marker='o')
plt.plot(results['Predicted'].values, label='Predicted', marker='x')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Test Data Index')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





