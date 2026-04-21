import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load engineered data
df = pd.read_csv('engineered_features.csv')
df['Close_Next'] = df['Close'].shift(-1)
df = df.dropna().reset_index(drop=True)

X = df.drop(columns=['Date', 'Close_Next'])
y = df['Close_Next']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📉 Mean Squared Error:", mse)
print("📈 R² Score:", r2)

# Save predictions
results = X_test.copy()
results['Actual_Close_Next'] = y_test.values
results['Predicted_Close_Next'] = y_pred
results.to_csv('model_predictions.csv', index=False)
print("✅ Predictions saved to model_predictions.csv")
