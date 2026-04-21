import pandas as pd
import numpy as np

# Generate dates (monthly) from Jan 2015 to Dec 2023
date_range = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')

# Generate random rainfall values (in mm)
rainfall = np.random.uniform(50, 300, size=len(date_range))

# Create DataFrame
rainfall_df = pd.DataFrame({
    'Date': date_range,
    'Rainfall': rainfall
})

# Save as CSV
rainfall_df.to_csv('rainfall_data.csv', index=False)

print("Dummy rainfall_data.csv created.")
