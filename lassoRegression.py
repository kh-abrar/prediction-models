import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv('FinalDataset.csv')

# Extract features from the 'Date' column
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Convert the target variable to numeric format by removing commas
data['Total Yield [kWh]'] = data['Total Yield [kWh]'].replace(',', '', regex=True).astype(float)

# Drop rows with NaN values in the target variable
data = data.dropna(subset=['Total Yield [kWh]'])

# Separate features and target
X = data[['Year', 'Month', 'Day', 'Highest Temp (C°)', 'Lowest Temp (C°)', 'Average Temp (C°)']]  # Include the new date-related features
y = data['Total Yield [kWh]']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Lasso Regression model
alpha = 0.1  # You can adjust this value
lasso = Lasso(alpha=alpha, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = lasso.predict(X_test_scaled)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Display warning message if the maximum number of iterations was reached
if lasso.n_iter_ == 10000:
    print("Warning: Lasso Regression did not converge. Consider adjusting parameters.")
