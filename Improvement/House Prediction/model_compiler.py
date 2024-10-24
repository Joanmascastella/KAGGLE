import torch
import torch.nn as nn
import torch.optim as optim
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
        x = self.output(x)  # Output is a single value (house price)
        return x

device = hf.get_device()

def define_parameters(model):
    model = model

    # Define the optimizer, learning rate, and loss function
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train model
    n_epochs = 100
    loss_list = []
    accuracy_list = []

    return model, optimizer, criterion, loss_list, accuracy_list, n_epochs


def compile_and_train_model(train_loader, test_loader, model, optimizer, criterion, loss_list, accuracy_list, n_epochs):
    print(f"train_loader type: {type(train_loader)}, content: {train_loader}")

    def train(n_epochs):
        for epoch in range(n_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0

            for x, y in train_loader:  # Assuming y is available in train_loader
                x, y = x.to(device), y.to(device)  # Move input and target to device

                # Forward pass
                z = model(x)  # Model prediction
                loss = criterion(z, y.view(-1, 1))  # Calculate loss (y may need reshaping)

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                running_loss += loss.item()  # Accumulate loss

            # Average loss for the epoch
            epoch_loss = running_loss / len(train_loader)
            loss_list.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}")

        # Perform Validation
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)  # Move input and target to device
                z = model(x)  # Model prediction

                # Calculate the accuracy
                predicted = z.view(-1)  # Reshape predictions
                total += y.size(0)  # Total number of samples
                # Compute mean absolute error for regression (you can use another metric if needed)
                correct += torch.sum(torch.abs(predicted - y.view(
                    -1)) < 0.5).item()  # Thresholding for correctness (adjust the threshold as needed)

        accuracy = correct / total  # Calculate accuracy
        accuracy_list.append(accuracy)  # Store accuracy for tracking
        print(f"Validation Accuracy: {accuracy:.4f}")

    train(n_epochs)  # Start training process
    return accuracy_list, loss_list