import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

# File paths
train_file = "/Users/joanmascastella/Desktop/CODE/KAGGLE/house-prices-advanced-regression-techniques/train.csv"
test_file = "test.csv"
sample_file = "sample_submission.csv"

# Load the data
train_df = pd.read_csv(train_file)

# Separate the features (X) and the target (y)
X = train_df.drop(columns=["SalePrice"])
y = train_df["SalePrice"].values

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Handle missing values by imputing them with the mean
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Standardize the feature matrix X
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Initialize the K-Nearest Neighbors Regressor with k=57
k = 6
neigh = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)

# Make predictions on the test set
yhat = neigh.predict(X_test)

# Calculate and print the Mean Squared Error and Mean Absolute Error
mse = mean_squared_error(y_test, yhat)
mae = mean_absolute_error(y_test, yhat)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Scatter plot of actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, yhat, c='blue', label='Predicted vs Actual', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - KNN Regressor')
plt.legend()
plt.show()

# Tuning the value of k
Ks = 20  # Maximum number of neighbors to test
mean_mse = np.zeros((Ks-1))
mean_mae = np.zeros((Ks-1))

# Loop through values of k from 1 to Ks
for n in range(1, Ks):
    neigh = KNeighborsRegressor(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_mse[n-1] = mean_squared_error(y_test, yhat)
    mean_mae[n-1] = mean_absolute_error(y_test, yhat)

# Plot MSE for different values of k
plt.figure(figsize=(8, 6))
plt.plot(range(1, Ks), mean_mse, 'g', label='MSE')
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Number of Neighbors (k)')
plt.title('K-Nearest Neighbors MSE by Number of Neighbors')
plt.legend()
plt.show()

# Plot MAE for different values of k
plt.figure(figsize=(8, 6))
plt.plot(range(1, Ks), mean_mae, 'r', label='MAE')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Number of Neighbors (k)')
plt.title('K-Nearest Neighbors MAE by Number of Neighbors')
plt.legend()
plt.show()

# Print the best k based on MSE and MAE
best_k_mse = mean_mse.argmin() + 1
best_k_mae = mean_mae.argmin() + 1
print(f"The lowest MSE is {mean_mse.min()} with k={best_k_mse}")
print(f"The lowest MAE is {mean_mae.min()} with k={best_k_mae}")