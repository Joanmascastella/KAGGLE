import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()

        #Encoder
        #Encoder deconstructs the data
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128,hidden_size)
        )

        #Decoder
        #Decoder reconstructs the data and measure its loss
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128,input_size),
            nn.Sigmoid()
        )

        #Forward Pass
        # Perfoms encoder and reconstruction
        def forward(self, x):
            latent_space = self.encoder(x)
            reconstructed = self.decoder(latent_space)
            return reconstructed, latent_space


def extract_features(train_data_loader, test_data_loader, input_size, hidden_size, learning_rate=0.0005, num_epochs=30):
    #Initializing autoencoder model
    autoencoder = Autoencoder(input_size = input_size, hidden_size = hidden_size)

    #Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    #Training Loop
    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0

        for data in train_data_loader:
            inputs, _ = data # For the autoencoder we don't need to pass the labels we just need the inputs
            inputs = inputs.to(torch.float32)

            #Forward Pass: reconstruct the input
            reconstructed, _ = autoencoder(inputs)

            #Compute reconstruction loss
            loss = criterion(reconstructed, inputs)

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_data_loader)}")

        # Feature extraction on both train and test datasets
        train_features = []
        test_features = []

        autoencoder.eval()

        #Extract Train Features
        with torch.no_grad():
            for data in train_data_loader:
                inputs, _ = data
                inputs = inputs.to(torch.float32)
                _, latent_space = autoencoder(inputs)
                train_features.append(latent_space)

        with torch.no_grad():
            for data in test_data_loader:
                inputs, _ = data
                inputs = inputs.to(torch.float32)
                _, latent_space = autoencoder(inputs)
                test_features.append(latent_space)

        #Conver the lists back to tensors
        train_features = torch.cat(train_features, dim=0).cpu().numpy()
        test_features = torch.cat(test_features, dim=0).cpu().numpy()

        return train_features, test_features



