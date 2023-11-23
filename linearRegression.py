import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

# Separate features and target
X = data[['Year', 'Month', 'Day', 'Highest Temp (C°)', 'Lowest Temp (C°)', 'Average Temp (C°)']]
y = data['Total Yield [kWh]']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Drop NaN values from both X_train and y_train
train_data = pd.concat([X_train, y_train], axis=1).dropna()
X_train = train_data[X_train.columns]
y_train = train_data[y_train.name]

# Verify the lengths after dropping NaN values
print(len(X_train), len(y_train))

# Create and train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# R-squared score (coefficient of determination)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
