import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertModel
import torch

# Load train and test data into their respective data frames
train_df = pd.read_csv('/Old/NLP/kaggle/train.csv')
test_df = pd.read_csv('/Old/NLP/kaggle/test.csv')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

def get_bert_embeddings(texts):
    """Get BERT embeddings for a list of texts."""
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    # Use the mean of the last hidden state as the embedding
    embeddings = output.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Generate BERT embeddings for train and test datasets
train_embeddings = get_bert_embeddings(train_df["text"].tolist())
test_embeddings = get_bert_embeddings(test_df["text"].tolist())

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

    for train_index, val_index in kf.split(train_embeddings):
        X_train, X_val = train_embeddings[train_index], train_embeddings[val_index]
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
best_model.fit(train_embeddings, train_df["target"])

# Predict on the test set
test_predictions = best_model.predict(test_embeddings)

# Prepare submission file
sample_submission = pd.read_csv("/Old/NLP/kaggle/sample_submission.csv")
sample_submission["target"] = test_predictions

# Print and save the submission file
print(sample_submission.head())
sample_submission.to_csv("/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/submission.csv", index=False)