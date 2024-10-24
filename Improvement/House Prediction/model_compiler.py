import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import helpful_functions as hf
from sklearn.model_selection import train_test_split
import config

# Define the fully connected network for house price prediction
class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)  # First layer (input to hidden)
        self.relu1 = nn.ReLU()  # Activation function

        self.fc2 = nn.Linear(128, 64)  # Second hidden layer
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 32)  # Third hidden layer
        self.relu3 = nn.ReLU()

        self.output = nn.Linear(32, 1)  # Output layer (single value for house price prediction)

        # Initialize weights using Kaiming (He) initialization for better convergence
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.output.weight, nonlinearity="relu")

    def forward(self, x):
        # Pass through the fully connected layers with ReLU activations
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = torch.clamp(self.output(x), min=0)
        return x

device = hf.get_device()

def define_parameters(model):
    model = model

    # Define the optimizer, learning rate, and loss function
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Use Mean Squared Error loss for regression

    # Train model
    n_epochs = 100
    loss_list = []
    accuracy_list = []

    return model, optimizer, criterion, loss_list, accuracy_list, n_epochs


device = hf.get_device()


# Custom RMSE criterion with log transformation
def rmse_log_loss(y_pred, y):
    # Clamp values to avoid taking log of zero or negative values
    y_pred = torch.clamp(y_pred, min=1e-6)
    y = torch.clamp(y, min=1e-6)

    # Take the log of predictions and true values
    y_pred_log = torch.log(y_pred)
    y_log = torch.log(y)

    # Calculate Mean Squared Error between log values
    mse_loss = F.mse_loss(y_pred_log, y_log)

    # Return Root Mean Squared Error (RMSE)
    return torch.sqrt(mse_loss)

def define_parameters(model):
    model = model

    # Define the optimizer, learning rate, and custom loss function
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = rmse_log_loss  # Use this custom loss for training

    n_epochs = 10
    loss_list = []
    mae_list = []  # Track Mean Absolute Error (MAE) for accuracy

    return model, optimizer, criterion, loss_list, mae_list, n_epochs


def compile_and_train_model(train_loader, test_loader, model, optimizer, criterion, loss_list, mae_list, n_epochs, submission_file_path):
    def train(n_epochs):
        for epoch in range(n_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            running_mae = 0.0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # Forward pass
                y_pred = model(x)
                loss = criterion(y_pred, y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate Mean Absolute Error (MAE) for accuracy tracking
                mae = torch.mean(torch.abs(y_pred - y)).item()
                running_mae += mae

            # Average loss and MAE for the epoch
            epoch_loss = running_loss / len(train_loader)
            epoch_mae = running_mae / len(train_loader)
            loss_list.append(epoch_loss)
            mae_list.append(epoch_mae)

            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}")

    # Train the model
    train(n_epochs)
    hf.plot_metrics_with_dual_axes(loss_list, mae_list)

    # Predict for the test data and save results for submission
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        for x in test_loader:
            x = x[0].to(device).float()  # Move input to device
            y_pred = model(x).cpu().numpy()
            predictions.extend(y_pred.flatten())  # Collect predictions

    # Create submission dataframe and save to CSV
    submission_df = pd.read_csv(submission_file_path)  # Read sample submission file
    submission_df['SalePrice'] = predictions  # Replace with predicted prices
    submission_df.to_csv('submission.csv', index=False)  # Save the predictions

    print("Predictions saved to submission.csv.")

    return mae_list, loss_list