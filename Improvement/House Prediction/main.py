from torch.utils.data import TensorDataset, DataLoader
import data as d
import feature_extraction as ft
import model_compiler as mc
import helpful_functions as hf
import torch

# File paths
train_file_path = './data/train.csv'
test_file_path = './data/test.csv'
submission_file_path = './data/sample_submission.csv'

# Device
device = hf.get_device()

def main():
    # Step 1: Data Loading and Cleaning
    print("Loading and cleaning data...")
    train_cleaned, test_cleaned, y_train = d.load_and_clean_data(train_file_path, test_file_path)
    print("Creating data loader objects...")
    train_data_loader, test_data_loader = d.get_data_loaders(train_cleaned, y_train, test_cleaned)

    # Step 2: Feature Extraction
    print("Extracting features using Autoencoder...")
    input_size = train_cleaned.shape[1]  # Number of input features
    hidden_size = 40  # Latent space size (tunable)

    # Extract features using the autoencoder
    train_features, test_features = ft.extract_features(train_data_loader, test_data_loader, input_size, hidden_size)

    # Step 3: Model Compilation, Training, and Testing
    print("Compiling and training model with extracted features...")
    # Use the number of features from the extracted features
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    input_size = train_features_tensor.shape[1]  # Get the number of features
    model = mc.HousePriceModel(input_size=input_size).to(device)
    model, optimizer, criterion, loss_list, accuracy_list, n_epochs = mc.define_parameters(model)

    # Prepare labels and test features
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

    # Create DataLoaders for train_features and test_features
    train_dataset = TensorDataset(train_features_tensor, y_train_tensor)  # Create TensorDataset for train
    test_dataset = TensorDataset(test_features_tensor)  # Create TensorDataset for test

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    results, loss = mc.compile_and_train_model(train_loader, test_loader, model, optimizer, criterion, loss_list,
                                               accuracy_list, n_epochs, submission_file_path)

    print("Model training completed.")
    print(f"Results: {results}")
    print(f"Loss: {loss}")

if __name__ == '__main__':
    main()

