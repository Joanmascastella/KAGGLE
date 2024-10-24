import torch
import torch.nn as nn
import torch.optim as optim
import helpful_functions as hf


# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size)  # Latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return reconstructed, latent_space


# Function to extract features using autoencoder
def extract_features(train_data_loader, test_data_loader, input_size, hidden_size, learning_rate=0.0005, num_epochs=120):
    device = hf.get_device()  # Get the appropriate device

    # Initialize the autoencoder model and move to device
    autoencoder = Autoencoder(input_size=input_size, hidden_size=hidden_size).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # To store the loss over epochs for plotting
    all_losses = []

    # Training loop
    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0

        for data in train_data_loader:
            inputs, _ = data
            inputs = inputs.to(device).float()  # Move data to device

            # Forward pass: reconstruct the input
            reconstructed, _ = autoencoder(inputs)

            # Compute reconstruction loss
            loss = criterion(reconstructed, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log and store loss
        epoch_loss = running_loss / len(train_data_loader)
        all_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Plot the training loss
    hf.autoencoder_plot_loss(all_losses)

    # Feature extraction on both train and test datasets
    train_features = []
    test_features = []

    autoencoder.eval()  # Set the model to evaluation mode

    # Extract features from train data
    with torch.no_grad():
        for data in train_data_loader:
            inputs, _ = data
            inputs = inputs.to(device).float()
            latent_space = autoencoder.encoder(inputs)
            train_features.append(latent_space.cpu())

    # Extract features from test data
    with torch.no_grad():
        for data in test_data_loader:
            inputs = data[0].to(device).float()  # Test data does not have labels
            latent_space = autoencoder.encoder(inputs)
            test_features.append(latent_space.cpu())

    # Convert feature lists into tensors
    train_features = torch.cat(train_features, dim=0)  # Concatenate all tensors in the list
    test_features = torch.cat(test_features, dim=0)    # Concatenate all tensors in the list

    # Return extracted features as tensors
    return train_features, test_features