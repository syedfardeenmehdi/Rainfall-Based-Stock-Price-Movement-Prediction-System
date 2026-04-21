import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data with lag features
data = pd.read_csv('merged_with_lags.csv')

# Define target and features
target = 'Close'
features = [col for col in data.columns if col not in ['Date', 'Close']]

X = data[features]
y = data[target]

# Train-test split (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📉 Mean Squared Error (Random Forest): {mse}")
print(f"📈 R² Score (Random Forest): {r2}")

# Optional: Save predictions
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred
results.to_csv('rf_model_predictions.csv', index=False)

print("✅ Predictions saved to rf_model_predictions.csv")
