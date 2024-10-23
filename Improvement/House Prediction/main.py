import data as d
import feature_extraction as ft
import model_compiler as mc

# File paths
train_file_path = './data/train.csv'
test_file_path = './data/test.csv'
submission_file_path = './data/sample_submission.csv'


def main():
    # Step 1: Data Loading and Cleaning
    print("Loading and cleaning data...")
    train_data, test_data = d.load_and_clean_data(train_file_path, test_file_path)

    # Step 2: Feature Extraction (e.g., Autoencoder)
    print("Extracting features...")
    train_features, test_features = ft.extract_features(train_data, test_data)

    # Step 3: Model Compilation, Training, and Testing
    print("Compiling and training model...")
    results, loss = mc.compile_and_train_model(train_features, test_features)

    # Return results and loss
    print("Model training completed.")
    print(f"Results: {results}")
    print(f"Loss: {loss}")


if __name__ == '__main__':
    main()