import data as d
import feature_extraction as ft
import model_compiler as mc
import config

# File paths
train_file_path = './data/train.csv'
test_file_path = './data/test.csv'
submission_file_path = './data/sample_submission.csv'

def main():
    # Step 1: Data Loading and Cleaning
    print("Loading and cleaning data...")
    train_cleaned, test_cleaned, y_train = d.load_and_clean_data(train_file_path, test_file_path)
    print("Creating data loader objects...")
    train_data_loader, test_data_loader = d.get_data_loaders(train_cleaned, y_train, test_cleaned)

    # Step 2: Feature Extraction
    print("Extracting features using Autoencoder...")
    input_size = train_cleaned.shape[1]  # Number of input features
    hidden_size = 64  # Latent space size (tunable)
    train_features, test_features = ft.extract_features(train_data_loader, test_data_loader, input_size, hidden_size)

    # Step 3: Model Compilation, Training, and Testing
    print("Compiling and training model with extracted features...")
    config.INPUT_SIZE = train_features.shape[1]
    results, loss = mc.compile_and_train_model(train_features, test_features)

    # Return results and loss
    print("Model training completed.")
    print(f"Results: {results}")
    print(f"Loss: {loss}")

if __name__ == '__main__':
    main()