import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset
import torch
from scipy.sparse import issparse

# Function to load, clean, and preprocess the data
def load_and_clean_data(train_data, test_data):
    # Load the data
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)

    # Separate features and target in the training data
    y_train = train_data['SalePrice']
    X_train = train_data.drop(['SalePrice'], axis=1)
    X_test = test_data

    # Identify numerical and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Pipelines for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('scaler', StandardScaler())  # Standardize numeric features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    # Combine both transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply transformations to train and test data
    X_train_cleaned = preprocessor.fit_transform(X_train)
    X_test_cleaned = preprocessor.transform(X_test)

    # Ensure shapes match
    print(f"Train Data Shape: {X_train_cleaned.shape}")
    print(f"Test Data Shape: {X_test_cleaned.shape}")

    return X_train_cleaned, X_test_cleaned, y_train


# Create DataLoader for PyTorch (optional)
def get_data_loaders(X_train_cleaned, y_train, X_test_cleaned, batch_size=32):
    # Check if the matrix is sparse and convert it to dense if necessary
    if issparse(X_train_cleaned):
        X_train_cleaned = X_train_cleaned.toarray()  # Convert to dense matrix
    if issparse(X_test_cleaned):
        X_test_cleaned = X_test_cleaned.toarray()  # Convert to dense matrix

    # Convert the data to PyTorch tensors
    train_tensor = TensorDataset(torch.tensor(X_train_cleaned, dtype=torch.float32),
                                 torch.tensor(y_train.values, dtype=torch.float32))
    test_tensor = TensorDataset(torch.tensor(X_test_cleaned, dtype=torch.float32))

    # Create DataLoader objects
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
