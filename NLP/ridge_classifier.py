import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load train and test data into their respective data frames
train_df = pd.read_csv('/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/train.csv')
test_df = pd.read_csv('/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/test.csv')

# Using TfidfVectorizer from sklearn to build vectors
vectorizer = TfidfVectorizer(max_features=20000)

# Vectorize the text data for train and test datasets
train_vectors = vectorizer.fit_transform(train_df["text"])
test_vectors = vectorizer.transform(test_df["text"])

# Convert sparse matrices to dense format for RidgeClassifier (optional)
train_vectors_np = train_vectors.toarray()
test_vectors_np = test_vectors.toarray()

# Define hyperparameters to test
param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Regularization strength for RidgeClassifier
}

best_f1 = 0
best_params = None

# Perform manual hyperparameter tuning
for alpha in param_grid['alpha']:
    print(f"Testing model with alpha={alpha}")

    # Perform cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=43)
    f1_scores = []

    for train_index, val_index in kf.split(train_vectors_np):
        X_train, X_val = train_vectors_np[train_index], train_vectors_np[val_index]
        y_train, y_val = train_df["target"].values[train_index], train_df["target"].values[val_index]

        # Build and train the RidgeClassifier model
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train, y_train)

        # Predict and calculate F1 score
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append(f1)

    # Calculate average F1 score for this set of hyperparameters
    average_f1 = np.mean(f1_scores)
    print(f"Average F1 Score: {average_f1}")

    # Update best parameters if current model is better
    if average_f1 > best_f1:
        best_f1 = average_f1
        best_params = {
            'alpha': alpha
        }

print(f"Best parameters: {best_params} with F1 Score: {best_f1}")

# Train the final model with the best hyperparameters
best_model = RidgeClassifier(alpha=best_params['alpha'])
best_model.fit(train_vectors_np, train_df["target"])

# Predict on the test set
test_predictions = best_model.predict(test_vectors_np)

# Prepare submission file
sample_submission = pd.read_csv("/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/sample_submission.csv")
sample_submission["target"] = test_predictions

# Print and save the submission file
print(sample_submission.head())
sample_submission.to_csv("/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/submission.csv", index=False)