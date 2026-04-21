import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('enhanced_data_for_rf.csv')
target = 'Close'
exclude = ['Date', 'Close']
features = [col for col in data.columns if col not in exclude]

X = data[features]
y = data[target]

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Define the model
rf = RandomForestRegressor(random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# Grid Search with TimeSeriesSplit
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

grid_search.fit(X, y)

# Best model
best_model = grid_search.best_estimator_

# Final prediction using 80/20 split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📉 Tuned Mean Squared Error: {mse}")
print(f"📈 Tuned R² Score: {r2}")
print(f"✅ Best Parameters: {grid_search.best_params_}")

# Save predictions
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred
results.to_csv('rf_model_predictions_tuned.csv', index=False)
print("✅ Predictions saved to 'rf_model_predictions_tuned.csv'")
