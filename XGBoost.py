import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('FinalDataset.csv')

# Extract features from the 'Date' column
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Convert the target variable to numeric format by removing commas
data['Total Yield [kWh]'] = data['Total Yield [kWh]'].replace(',', '', regex=True).astype(float)

# Feature engineering: Adding lag features
data['Previous_Day_Yield'] = data['Total Yield [kWh]'].shift(1)

# Drop rows with missing target values
data_clean = data.dropna(subset=['Total Yield [kWh]'])
X_clean = data_clean[['Year', 'Month', 'Day', 'Highest Temp (C°)', 'Lowest Temp (C°)', 'Average Temp (C°)', 'Previous_Day_Yield']]
y_clean = data_clean['Total Yield [kWh]']

# Check for infinity or extremely large values in the target variable
print("Contains Infinity:", np.any(np.isinf(y_clean)))
print("Max Value:", np.max(y_clean))

# Split the clean data into training and testing sets
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)

# Create and train the XGBoost model
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train_clean, y_train_clean)

# Make predictions on the test set
y_pred = xgb.predict(X_test_clean)

# Evaluate the model's performance
mse = mean_squared_error(y_test_clean, y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
