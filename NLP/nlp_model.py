import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Load train and test data into their respective data frames
train_df = pd.read_csv('/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/train.csv')
test_df = pd.read_csv('/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/test.csv')

# Building vectors using Keras TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(max_tokens=20000, output_mode='tf-idf')
text_ds = tf.data.Dataset.from_tensor_slices(train_df["text"].values).batch(32)
vectorizer.adapt(text_ds)

# Vectorize the text data
train_vectors = vectorizer(np.array(train_df["text"]))
test_vectors = vectorizer(np.array(test_df["text"]))

# Convert TensorFlow tensor to a NumPy array
train_vectors_np = train_vectors.numpy()
test_vectors_np = test_vectors.numpy()

def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(train_vectors_np.shape[1],)),
        
        # First hidden layer
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Second hidden layer
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Optional Dropout layer for regularization
        tf.keras.layers.Dropout(0.5),
        
        # Third hidden layer
        tf.keras.layers.Dense(32, activation='relu'),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define hyperparameters to test
param_grid = {
    'epochs': [10, 20, 50],
    'batch_size': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1]
}

best_f1 = 0
best_params = None

# Perform manual hyperparameter tuning
for epochs in param_grid['epochs']:
    for batch_size in param_grid['batch_size']:
        for learning_rate in param_grid['learning_rate']:
            print(f"Testing model with epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")

            # Perform cross-validation
            kf = KFold(n_splits=3, shuffle=True, random_state=43)
            f1_scores = []

            for train_index, val_index in kf.split(train_vectors_np):
                X_train, X_val = train_vectors_np[train_index], train_vectors_np[val_index]
                y_train, y_val = train_df["target"].values[train_index], train_df["target"].values[val_index]

                # Build and train the model
                model = build_model(learning_rate)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                # Predict and calculate F1 score
                y_pred = (model.predict(X_val) > 0.5).astype(int)
                f1 = f1_score(y_val, y_pred)
                f1_scores.append(f1)

            # Calculate average F1 score for this set of hyperparameters
            average_f1 = np.mean(f1_scores)
            print(f"Average F1 Score: {average_f1}")

            # Update best parameters if current model is better
            if average_f1 > best_f1:
                best_f1 = average_f1
                best_params = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }

print(f"Best parameters: {best_params} with F1 Score: {best_f1}")

# Train the final model with the best hyperparameters
best_model = build_model(best_params['learning_rate'])
# best_model = build_model(0.01)
best_model.fit(train_vectors_np, train_df["target"], epochs=best_params['epochs'], batch_size=best_params['batch_size'])
# best_model.fit(train_vectors_np, train_df["target"], epochs=10, batch_size=128)

# Predict on the test set
test_predictions = (best_model.predict(test_vectors_np) > 0.5).astype(int)

# Prepare submission file
sample_submission = pd.read_csv("/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/sample_submission.csv")
sample_submission["target"] = test_predictions

# Print and save the submission file
print(sample_submission.head())
sample_submission.to_csv("/Users/joanmascastella/Desktop/CODE/backup/NLP/kaggle/submission.csv", index=False)